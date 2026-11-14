"""Incorporación local reproducible de un PDF aportado por el usuario."""
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from radiology_rag.artifacts import artifact_stage
from radiology_rag.ingestion.download import validate_pdf
from radiology_rag.paths import within


def import_local(root: Path, doc: dict, source: Path, *, authorization_note: str) -> dict:
    if not doc["rights"]["auto_download"] and not authorization_note.strip():
        raise PermissionError("Registrar la cobertura de uso confirmada para este documento.")
    source = source.resolve(strict=True)
    identity = validate_pdf(source, doc)
    target = within(root, doc["local_path"])
    if not target.is_relative_to((root / "data/raw").resolve()):
        raise ValueError("Los originales se incorporan dentro de data/raw/.")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(".import.lock")
    try:
        lock_stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError("Ya hay una incorporación en curso para este documento.") from None
    manifest_path = target.with_suffix(".source.json")
    with lock_stream:
        try:
            if target.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (validate_pdf(target, doc) != identity
                        or existing["sha256"] != identity["sha256"]
                        or existing["document_id"] != doc["id"]
                        or existing["version"] != doc["version"]):
                    raise ValueError("Ya existe otra identidad o manifiesto para este documento.")
                return existing
            record = {
                "document_id": doc["id"], "version": doc["version"],
                "source_url": doc["source_url"], "local_path": doc["local_path"],
                "acquisition_method": "user_provided_local_file",
                "provided_filename": source.name,
                "imported_at": datetime.now(UTC).isoformat(),
                "rights_basis": authorization_note.strip() or doc["rights"]["license"],
                **identity,
            }
            with artifact_stage(target.parent, prefix=".import-") as directory:
                staging = Path(directory).resolve()
                if not staging.is_relative_to((root / "data/raw").resolve()):
                    raise ValueError("El staging de incorporación sale del área de originales.")
                staged_pdf = staging / target.name
                shutil.copyfile(source, staged_pdf)
                if validate_pdf(staged_pdf, doc) != identity:
                    raise ValueError("El archivo cambió durante la incorporación.")
                staged_manifest = staging / manifest_path.name
                staged_manifest.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                staged_pdf.rename(target)
                try:
                    staged_manifest.rename(manifest_path)
                except OSError:
                    target.unlink()
                    raise
            return record
        finally:
            lock_stream.close()
            lock.unlink(missing_ok=True)
