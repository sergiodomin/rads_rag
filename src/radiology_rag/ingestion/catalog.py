import hashlib
import json
from pathlib import Path

from radiology_rag.paths import within


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog["schema_version"] != 1:
        raise ValueError("Versión de esquema no soportada.")
    ids = set()
    for doc in catalog["documents"]:
        if doc["id"] in ids:
            raise ValueError("Identificador documental duplicado.")
        ids.add(doc["id"])
        for key in ("title", "version", "source_url", "rights", "local_path", "status"):
            if key not in doc:
                raise ValueError(f"Falta {key} en {doc['id']}.")
        within(path.parent.parent.parent, doc["local_path"])
    return catalog


def local_documents(root: Path) -> list[dict]:
    catalog = load_catalog(root / "data/catalog/documents.v1.json")
    return [d for d in catalog["documents"] if within(root, d["local_path"]).is_file()]
