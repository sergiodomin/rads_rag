"""Descarga explícita del tokenizer fijado; nunca descarga pesos ni lee .env."""
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within


def main():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "data/catalog/tokenizers.v1.json").read_text(encoding="utf-8"))[
        "tokenizers"
    ][0]
    target = within(root, spec["local_path"])
    if target.exists():
        digest = sha256_file(target)
        if spec.get("sha256") and digest != spec["sha256"]:
            raise ValueError("El tokenizer local no coincide con el catálogo.")
        print("Tokenizer ya disponible; SHA-256:", digest)
        return
    url = (
        f"https://huggingface.co/{spec['repository']}/resolve/"
        f"{spec['revision']}/{spec['filename']}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            size = 0
            with partial.open("wb") as stream:
                for block in response.iter_bytes():
                    size += len(block)
                    if size > 50 * 1024 * 1024:
                        raise ValueError("Tokenizer mayor de 50 MiB; revisar antes de incorporar.")
                    stream.write(block)
        digest = sha256_file(partial)
        if spec.get("sha256") and digest != spec["sha256"]:
            raise ValueError("La descarga no coincide con el hash fijado.")
        Tokenizer.from_file(str(partial))
        partial.replace(target)
        manifest = {
            **spec, "download_url": url, "sha256": digest, "bytes": size,
            "retrieved_at": datetime.now(UTC).isoformat(), "model_weights_downloaded": False,
        }
        target.with_suffix(".source.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
