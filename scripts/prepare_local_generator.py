"""Descarga explícita del generador local y su tokenizer, con identidades fijadas."""
import json

import httpx

from radiology_rag.config import Settings
from radiology_rag.generation.local_runtime import ensure_local_runtime
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import project_root, within


def main():
    root = project_root()
    spec = json.loads((root / "data/catalog/local_generator.v1.json").read_text(encoding="utf-8"))
    settings = Settings(base_url="http://127.0.0.1:11434", model=spec["model"], no_cloud=True,
                        num_ctx=spec["num_ctx"], num_predict=spec["num_predict"])
    ensure_local_runtime(root, settings)
    with httpx.Client(base_url=settings.base_url, timeout=120) as client:
        progress = {}
        with client.stream("POST", "/api/pull", json={"model": spec["model"], "stream": True}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                record = json.loads(line)
                if "error" in record:
                    raise RuntimeError(record["error"])
                digest = record.get("digest", "")
                done, total = record.get("completed", 0), record.get("total", 0)
                if not total or done - progress.get(digest, 0) >= 64 * 1024**2 or done == total:
                    print(record["status"], f"{done / 1024**2:.0f} MiB" if total else "",
                          flush=True)
                    progress[digest] = done
        models = client.get("/api/tags").json()["models"]
        match = next(m for m in models if m["name"] == spec["model"])
        if match["digest"] != spec["ollama_digest"]:
            raise ValueError("El tag del generador cambió; revisar el catálogo antes de usarlo.")
    tokenizer = spec["tokenizer"]
    folder = within(root, tokenizer["path"])
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for item in tokenizer["files"]:
            path = within(folder, item["path"])
            if not path.is_file():
                url = (f"https://huggingface.co/{tokenizer['repository']}/resolve/"
                       f"{tokenizer['revision']}/{item['path']}")
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
            if sha256_file(path) != item["sha256"]:
                raise ValueError("El tokenizer local no coincide con la revisión fijada.")
    print("Generador local y tokenizer verificados.", flush=True)


if __name__ == "__main__":
    main()
