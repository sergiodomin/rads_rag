"""Descarga modelos públicos por revisión y verifica sus archivos antes de publicarlos."""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within


def prepare(root: Path, spec: dict):
    target = within(root, spec["local_path"])
    target.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30)) as client:
        for item in spec["files"]:
            path = within(target, item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = item.get("sha256")
            if path.exists():
                digest = sha256_file(path)
                if path.stat().st_size != item["bytes"] or (expected and digest != expected):
                    raise ValueError(f"Archivo local modificado: {path.name}")
            else:
                partial = path.with_suffix(path.suffix + ".part")
                url = (f"https://huggingface.co/{spec['repository']}/resolve/"
                       f"{spec['revision']}/{item['path']}")
                try:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        size = 0
                        with partial.open("wb") as stream:
                            for block in response.iter_bytes(chunk_size=1024 * 1024):
                                stream.write(block)
                                size += len(block)
                                if size > item["bytes"]:
                                    raise ValueError("La descarga supera el tamaño catalogado.")
                    digest = sha256_file(partial)
                    if size != item["bytes"] or (expected and digest != expected):
                        raise ValueError(f"Identidad descargada incorrecta: {item['path']}")
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
            records.append({**item, "sha256": digest})
            print(f"{spec['id']}: verificado {item['path']} ({item['bytes']:,} bytes)", flush=True)
    manifest_path = target / ".source.json"
    record = {"repository": spec["repository"], "revision": spec["revision"],
              "license": spec["license"], "files": records,
              "prepared_at": datetime.now(UTC).isoformat()}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing[key] != record[key] for key in ("repository", "revision", "files")):
            raise ValueError("El manifiesto existente no coincide con los archivos verificados.")
    else:
        temporary = target / ".source.json.part"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog_path = root / "data/catalog/embedding_models.v1.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ids = args.model_ids or [m["id"] for m in catalog["models"]]
    for model_id in ids:
        spec = next((m for m in catalog["models"] if m["id"] == model_id), None)
        if spec is None:
            parser.error(f"Modelo desconocido: {model_id}")
        prepare(root, spec)


if __name__ == "__main__":
    main()
