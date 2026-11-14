import json

from pypdf import PdfWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from radiology_rag.ingestion import pipeline
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.chunking import TokenCounter


def test_processing_without_private_annotations_does_not_invent_evaluation(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    pdf = tmp_path / "data/raw/unit/manual.pdf"
    pdf.parent.mkdir(parents=True)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf.open("wb") as stream:
        writer.write(stream)
    digest = sha256_file(pdf)
    write(pdf.with_suffix(".source.json"), {
        "sha256": digest, "document_id": "unit", "version": "1",
    })
    doc = {"id": "unit", "title": "Synthetic test", "version": "1", "source_url": "unit-test",
           "rights": {}, "status": "test", "local_path": "data/raw/unit/manual.pdf",
           "role": "test", "processing": {"layout_path": "data/catalog/layout.json",
                                            "reference_path": "data/evaluations/private.json"}}
    write(tmp_path / "data/catalog/documents.v1.json", {
        "schema_version": 1, "active_document_id": "unit", "documents": [doc],
    })
    write(tmp_path / "data/catalog/layout.json", {"pages": [{"page_number": 1, "page_label": "1"}]})
    (tmp_path / "uv.lock").write_text("unit-test-lock", encoding="utf-8")
    implementation = tmp_path / "src/radiology_rag/evaluation/parsing.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# synthetic fingerprint fixture", encoding="utf-8")
    block = {"text": "synthetic alpha beta gamma", "block_id": "b1", "kind": "prose",
             "group_id": None, "document_id": "unit", "document_version": "1",
             "document_role": "test", "source_url": "unit-test", "source_sha256": digest,
             "parser_fingerprint": "test", "page_number": 1, "page_label": "1",
             "region_id": "p1", "bbox": [0, 0, 100, 100], "section_path": ["Test"],
             "unresolved_objects": []}
    monkeypatch.setattr(pipeline, "parse_regions", lambda *_: (
        [block], [], [{"page_number": 1, "decorative_images": []}],
    ))
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    counter = TokenCounter(tokenizer, {"passage_prefix": "passage: ", "encoder_max_tokens": 512,
                                      "revision": "unit", "sha256": "unit", "repository": "unit"})
    monkeypatch.setattr(TokenCounter, "from_project", classmethod(lambda *_: counter))
    report = pipeline.process_document(tmp_path)
    assert report["review_status"] == "not_provided_no_reference_evaluation"
    assert report["selected_characters"] == len(block["text"])
    assert len(report["metrics"]) == 9
    folder = tmp_path / f"data/processed/unit/02_chunking/runs/{report['run_id']}"
    assert not (folder / "parsing_reference_comparison.csv").exists()
    assert not report["encoder_executed"] and report["llm_calls"] == 0
