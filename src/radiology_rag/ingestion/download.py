import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pypdf import PdfReader

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within

ALLOWED_HOSTS = {
    "link.springer.com", "link.springernature.com", "edge.sitecorecloud.io",
    "pmc-oa-opendata.s3.amazonaws.com",
}
MAX_BYTES = 50 * 1024 * 1024


def validate_pdf(path: Path, doc: dict) -> dict:
    """Valida el archivo que se va a incorporar y devuelve su identidad física."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError("Documento mayor de 50 MiB: revisar manualmente.")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("La fuente no devolvió un PDF; no se guardará HTML como PDF.")
    digest = sha256_file(path)
    if doc.get("expected_sha256") and digest != doc["expected_sha256"]:
        raise ValueError("Cambió el SHA-256 de la fuente: revisar antes de incorporar.")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("PDF cifrado: incorporación manual necesaria.")
    pages = len(reader.pages)
    if doc.get("expected_pages") is not None and pages != doc["expected_pages"]:
        raise ValueError("Cambió el número de páginas: revisar la versión antes de incorporar.")
    return {"sha256": digest, "bytes": size, "pages": pages}


def acquire(root: Path, doc: dict, *, authorization_note: str = "", client=None) -> dict:
    """Descarga atómica. El catálogo no confiere permisos: registrar la base de uso."""
    if not doc["rights"]["auto_download"] and not authorization_note.strip():
        raise PermissionError("Fuente restringida: incorporar autorización explícita primero.")
    target = within(root, doc["local_path"])
    if not target.is_relative_to((root / "data/raw").resolve()):
        raise ValueError("Las fuentes sólo se guardan en data/raw/.")
    if target.exists():
        raise FileExistsError("Ya existe: no se sobrescribe una versión documental.")
    url = urlsplit(doc["download_url"])
    if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
        raise ValueError("La descarga requiere un host oficial HTTPS aprobado.")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pdf.part")
    own_client = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=False)
    try:
        with client.stream("GET", doc["download_url"]) as response:
            response.raise_for_status()
            size = 0
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Documento mayor de 50 MiB: revisar manualmente.")
                    stream.write(chunk)
        identity = validate_pdf(partial, doc)
        record = {
            "document_id": doc["id"], "version": doc["version"],
            "source_url": doc["source_url"], "download_url": doc["download_url"],
            "retrieved_at": datetime.now(UTC).isoformat(),
            **identity,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "rights_basis": authorization_note.strip() or doc["rights"]["license"],
            "local_path": doc["local_path"],
        }
        partial.replace(target)
        target.with_suffix(".source.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return record
    finally:
        partial.unlink(missing_ok=True)
        if own_client:
            client.close()
