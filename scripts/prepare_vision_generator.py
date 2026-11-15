"""Preparación explícita del modelo visual y su tokenizer; inferencia separada."""

import json
from pathlib import Path

import httpx

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within


def main():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "data/catalog/vision_generator.v1.json").read_text())
    with httpx.Client(base_url="http://127.0.0.1:11434", timeout=120) as client:
        with client.stream(
            "POST", "/api/pull", json={"model": spec["model"], "stream": True}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if "error" in json.loads(line):
                    raise ValueError("No se pudo preparar el modelo visual.")
        models = client.get("/api/tags").raise_for_status().json()["models"]
        match = next(m for m in models if m["name"] == spec["model"])
        if match["digest"] != spec["ollama_digest"]:
            raise ValueError("Cambió el tag visual: revisar su digest antes de usarlo.")
    tokenizer = spec["tokenizer"]
    folder = within(root, tokenizer["path"])
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for item in tokenizer["files"]:
            path = within(folder, item["path"])
            if not path.exists():
                url = (
                    f"https://huggingface.co/{tokenizer['repository']}/resolve/"
                    f"{tokenizer['revision']}/{item['path']}"
                )
                data = client.get(url).raise_for_status().content
                if len(data) != item["bytes"]:
                    raise ValueError("Tamaño de tokenizer inesperado.")
                path.write_bytes(data)
            if sha256_file(path) != item["sha256"]:
                raise ValueError("El tokenizer visual ha cambiado.")
    print("Modelo visual y tokenizer verificados.")


if __name__ == "__main__":
    main()
