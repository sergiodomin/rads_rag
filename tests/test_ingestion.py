import io
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter

from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.download import acquire
from radiology_rag.ingestion.local import import_local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def doc():
    catalog = load_catalog(ROOT / "data/catalog/documents.v1.json")
    record = deepcopy(next(d for d in catalog["documents"] if d["id"] == "esur-2012"))
    record.pop("expected_sha256", None)  # Los PDFs unitarios no son el documento descargado.
    return record


def test_restricted_document_cannot_download_without_permission(tmp_path, doc):
    doc["rights"]["auto_download"] = False
    with pytest.raises(PermissionError):
        acquire(tmp_path, doc)
    assert not list(tmp_path.rglob("*.pdf"))


def test_acquisition_records_bytes_hash_and_does_not_overwrite(tmp_path, doc):
    # PDF mínimo de prueba, no es un documento radiológico ni un resultado experimental.
    buffer = io.BytesIO()
    writer = PdfWriter()
    for _ in range(doc["expected_pages"]):
        writer.add_blank_page(width=200, height=200)
    writer.write(buffer)
    data = buffer.getvalue()
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=data, headers={"ETag": "unit-test-etag"})
    )) as client:
        record = acquire(tmp_path, doc, client=client)
        with pytest.raises(FileExistsError):
            acquire(tmp_path, doc, client=client)
    target = tmp_path / doc["local_path"]
    assert record["sha256"] == sha256_file(target)
    assert record["bytes"] == len(data) and record["pages"] == 12
    assert json.loads(target.with_suffix(".source.json").read_text(encoding="utf-8")) == record


def test_html_is_rejected_and_partial_file_removed(tmp_path, doc):
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"<html>login</html>")
    )) as client:
        with pytest.raises(ValueError, match="no devolvió un PDF"):
            acquire(tmp_path, doc, client=client)
    assert not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob("*.pdf"))


def test_changed_page_count_is_not_silently_accepted(tmp_path, doc):
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=buffer.getvalue())
    )) as client:
        with pytest.raises(ValueError, match="número de páginas"):
            acquire(tmp_path, doc, client=client)
    assert not list(tmp_path.rglob("*.pdf"))


def test_changed_source_hash_is_not_silently_accepted(tmp_path, doc):
    doc["expected_sha256"] = "0" * 64
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"%PDF-1.4 test data")
    )) as client:
        with pytest.raises(ValueError, match="SHA-256"):
            acquire(tmp_path, doc, client=client)
    assert not list(tmp_path.rglob("*.pdf"))


@pytest.mark.parametrize("url", ["http://link.springer.com/file.pdf", "https://untrusted.test/a.pdf"])
def test_untrusted_download_host_is_rejected(tmp_path, doc, url):
    doc["download_url"] = url
    with pytest.raises(ValueError, match="host oficial"):
        acquire(tmp_path, doc)


@pytest.mark.parametrize("path", ["../escape.pdf", "docs/overwrite.pdf"])
def test_download_target_stays_in_raw(tmp_path, doc, path):
    doc["local_path"] = path
    with pytest.raises(ValueError):
        acquire(tmp_path, doc)


def local_pdf(path, doc, width=200):
    writer = PdfWriter()
    for _ in range(doc["expected_pages"]):
        writer.add_blank_page(width=width, height=200)
    writer.write(path)
    return path


def test_local_import_records_origin_preserves_source_and_is_idempotent(tmp_path, doc):
    doc["rights"]["auto_download"] = False
    source = local_pdf(tmp_path / "provided.pdf", doc)
    original = source.read_bytes()
    record = import_local(tmp_path, doc, source, authorization_note="User confirmed local use")
    assert record["acquisition_method"] == "user_provided_local_file"
    assert "download_url" not in record and "retrieved_at" not in record
    assert record["sha256"] == sha256_file(source)
    assert source.read_bytes() == original
    assert import_local(tmp_path, doc, source, authorization_note="Confirmed") == record
    assert not list(tmp_path.rglob("*.lock"))


def test_local_import_requires_usage_record_for_restricted_source(tmp_path, doc):
    doc["rights"]["auto_download"] = False
    source = local_pdf(tmp_path / "provided.pdf", doc)
    with pytest.raises(PermissionError):
        import_local(tmp_path, doc, source, authorization_note="")
    assert not (tmp_path / doc["local_path"]).exists()


def test_local_import_rejects_changed_hash_before_writing(tmp_path, doc):
    source = local_pdf(tmp_path / "provided.pdf", doc)
    doc["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        import_local(tmp_path, doc, source, authorization_note="Confirmed")
    assert not (tmp_path / "data/raw").exists()


def test_local_import_does_not_replace_an_existing_identity(tmp_path, doc):
    source = local_pdf(tmp_path / "provided.pdf", doc)
    record = import_local(tmp_path, doc, source, authorization_note="Confirmed")
    different = local_pdf(tmp_path / "other.pdf", doc, width=300)
    with pytest.raises(ValueError, match="otra identidad"):
        import_local(tmp_path, doc, different, authorization_note="Confirmed")
    assert sha256_file(tmp_path / doc["local_path"]) == record["sha256"]


def test_local_import_keeps_another_import_lock(tmp_path, doc):
    source = local_pdf(tmp_path / "provided.pdf", doc)
    target = tmp_path / doc["local_path"]
    target.parent.mkdir(parents=True)
    lock = target.with_suffix(".import.lock")
    lock.write_text("another process", encoding="utf-8")
    with pytest.raises(FileExistsError):
        import_local(tmp_path, doc, source, authorization_note="Confirmed")
    assert lock.read_text(encoding="utf-8") == "another process"
    assert not target.exists()


def test_failed_local_copy_does_not_publish_or_leave_owned_lock(tmp_path, doc, monkeypatch):
    source = local_pdf(tmp_path / "provided.pdf", doc)

    def fail_copy(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("radiology_rag.ingestion.local.shutil.copyfile", fail_copy)
    with pytest.raises(OSError, match="simulated"):
        import_local(tmp_path, doc, source, authorization_note="Confirmed")
    assert not (tmp_path / doc["local_path"]).exists()
    assert not list((tmp_path / "data/raw").rglob("*.source.json"))
    assert not list(tmp_path.rglob("*.lock"))
