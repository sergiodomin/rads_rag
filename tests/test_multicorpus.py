import json

import numpy as np
import pytest

from radiology_rag.chat import ChatEngine
from radiology_rag.config import Settings
from radiology_rag.ingestion.catalog import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_indices(root, *, incompatible=False):
    documents = []
    for i, system in enumerate(("PI-RADS", "BI-RADS")):
        identifier = f"unit-{i}"
        pdf = root / f"data/raw/{identifier}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"unit-pdf-{i}".encode())
        documents.append({"id": identifier, "title": identifier, "version": "test",
                          "system": system, "source_url": "https://example.test",
                          "rights": {}, "status": "test",
                          "local_path": pdf.relative_to(root).as_posix()})
        index, report = root / f"data/indices/{identifier}", root / f"data/runs/{identifier}"
        folder = index / "e5-base/structural_512"
        folder.mkdir(parents=True)
        rows = [{"chunk_id": identifier, "document_id": identifier, "text": "unit evidence",
                 "parent_chunk_id": identifier, "pages": [1], "spans": [{"page_label": "1"}]}]
        (folder / "passages.jsonl").write_text(json.dumps(rows[0]) + "\n")
        np.save(folder / "embeddings.npy", np.eye(2, dtype=np.float32)[i:i + 1])
        inputs = {"models": [{"id": "e5-base", "dimension": 2,
                              "revision": "different" if incompatible and i else "test"}],
                  "collections": ["structural_512"], "source_sha256": sha256_file(pdf)}
        write(index / "manifest.json", {"run_id": identifier, "inputs": inputs,
              "artifact_sha256": {p.relative_to(index).as_posix(): sha256_file(p)
                                  for p in folder.iterdir()}})
        write(report / "manifest.json", {"run_id": identifier, "status": "complete",
              "inputs": inputs, "index_manifest_sha256": sha256_file(index / "manifest.json"),
              "artifact_sha256": {}})
        write(root / f"data/runs/{identifier}/03_embeddings/latest.json", {
            "run_id": identifier, "path": report.relative_to(root).as_posix(),
            "index_path": index.relative_to(root).as_posix()})
    write(root / "data/catalog/documents.v1.json", {
        "schema_version": 1, "active_document_id": "unit-0",
        "chat_document_ids": ["unit-0", "unit-1"], "documents": documents})


def test_loader_merges_only_compatible_vectors_and_preserves_row_alignment(tmp_path):
    setup_indices(tmp_path)
    engine = ChatEngine(tmp_path, Settings())
    rows, matrix = engine.collections["e5-base", "structural_512"]
    assert [row["system"] for row in rows] == ["PI-RADS", "BI-RADS"]
    np.testing.assert_array_equal(matrix, np.eye(2))
    assert len(engine.status()["documents"]) == 2
    (tmp_path / "data/raw/unit-1.pdf").write_bytes(b"modified")
    with pytest.raises(ValueError, match="PDF"):
        ChatEngine(tmp_path, Settings())


def test_loader_rejects_mixing_different_encoder_revisions(tmp_path):
    setup_indices(tmp_path, incompatible=True)
    with pytest.raises(ValueError, match="revisiones"):
        ChatEngine(tmp_path, Settings())
