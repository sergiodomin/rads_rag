"""Ejecuta el procesamiento documental y conserva cada resultado completo por huella."""
import csv
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from radiology_rag.artifacts import artifact_stage
from radiology_rag.evaluation.parsing import evidence_preservation, extraction_scores
from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.chunking import (
    TokenCounter,
    chunk_metrics,
    fixed_chunks,
    flatten_blocks,
    structural_chunks,
)
from radiology_rag.ingestion.download import validate_pdf
from radiology_rag.ingestion.parsing import fingerprint, parse_regions
from radiology_rag.paths import within


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def process_document(root: Path, document_id: str | None = None) -> dict:
    catalog = load_catalog(root / "data/catalog/documents.v1.json")
    document_id = document_id or catalog.get("active_document_id", "esur-2012")
    document = next((d for d in catalog["documents"] if d["id"] == document_id), None)
    if document is None:
        raise ValueError("Documento desconocido.")
    pdf_path = within(root, document["local_path"])
    identity = validate_pdf(pdf_path, document)
    source = json.loads(pdf_path.with_suffix(".source.json").read_text(encoding="utf-8"))
    if (source["sha256"] != identity["sha256"] or source["document_id"] != document_id
            or source["version"] != document["version"]):
        raise ValueError("El manifiesto de procedencia no corresponde al documento.")
    processing = document["processing"]
    layout = json.loads(within(root, processing["layout_path"]).read_text(encoding="utf-8"))
    reference_path = within(root, processing["reference_path"])
    references = (json.loads(reference_path.read_text(encoding="utf-8"))
                  if reference_path.is_file() else {
                      "source_sha256": identity["sha256"], "regions": [], "evidence_cases": [],
                      "status": "not_provided_no_reference_evaluation",
                  })
    if references["source_sha256"] != identity["sha256"]:
        raise ValueError("Las microreferencias pertenecen a otra copia documental.")
    counter = TokenCounter.from_project(root)
    environment = {"python": platform.python_version(), **{
        package: version(package) for package in ("tokenizers", "pypdf", "pdfplumber", "pandas")
    }}
    implementation = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in [root / "uv.lock", *sorted((root / "src/radiology_rag/ingestion").glob("*.py")),
                     root / "src/radiology_rag/evaluation/parsing.py"]
    }
    inputs = {
        "document_id": document_id, "document_version": document["version"],
        "source_sha256": identity["sha256"], "layout_fingerprint": fingerprint(layout),
        "references_fingerprint": fingerprint(references),
        "tokenizer_sha256": counter.spec["sha256"], "tokenizer_revision": counter.spec["revision"],
        "environment": environment, "implementation_sha256": implementation,
        "budgets": [256, 512, 768], "overlaps": [0, 0.15],
        "semantic": {"executed": False, "reason": "document_processing_only"},
    }
    run_id = fingerprint(inputs)[:16]
    output = root / "data/processed" / document_id / "02_chunking"
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    lock = output / ".process.lock"
    try:
        lock_stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError("Ya hay un procesamiento en curso para este documento.") from None
    try:
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            for name, digest in existing["artifact_sha256"].items():
                if sha256_file(within(destination, name)) != digest:
                    raise ValueError(f"Resultado modificado en la ejecución {run_id}: {name}")
            _update_latest(root, output, destination, run_id)
            return existing
        started = time.perf_counter()
        blocks, objects, audit = parse_regions(pdf_path, document, layout)
        text, located = flatten_blocks(blocks)
        if not text.strip():
            raise ValueError("El mapa no produjo texto para los experimentos.")
        reader = PdfReader(pdf_path)
        labels = {p["page_number"]: p.get("page_label", reader.page_labels[p["page_number"] - 1])
                  for p in layout["pages"]}
        pages = [{
            "document_id": document_id, "document_version": document["version"],
            "document_role": document["role"], "source_sha256": identity["sha256"],
            "source_url": document["source_url"], "page_number": index + 1,
            "page_label": labels[index + 1], "pdf_page_label": reader.page_labels[index],
            "text": page.extract_text() or "", "parser": "pypdf.default",
            "parser_version": version("pypdf"),
        } for index, page in enumerate(reader.pages)]
        comparison = []
        with pdfplumber.open(pdf_path) as pdf:
            for region in references["regions"]:
                crop = pdf.pages[region["page_number"] - 1].within_bbox(tuple(region["bbox"]))
                for label, options in [("default", {}), ("x_tolerance=1", {"x_tolerance": 1})]:
                    actual = crop.extract_text(**options) or ""
                    comparison.append({"region_id": region["id"], "page": region["page_number"],
                                       "kind": region["kind"], "parser_config": label,
                                       **extraction_scores(region["expected_text"], actual),
                                       "expected": region["expected_text"], "observed": actual})
        experiments, metrics, evidence = {}, [], []
        for budget in inputs["budgets"]:
            for overlap in inputs["overlaps"]:
                name = f"fixed_{budget}_{int(overlap * 100)}"
                experiments[name] = fixed_chunks(
                    text, located, counter, budget=budget, overlap=overlap
                )
            experiments[f"structural_{budget}"] = structural_chunks(
                text, located, counter, budget=budget
            )
        for name, chunks in experiments.items():
            metric = {"experiment": name, **chunk_metrics(chunks, text, located)}
            if metric["coverage_of_selected_text"] != 1:
                raise ValueError(f"El chunker perdió texto seleccionado: {name}")
            if len({c["chunk_id"] for c in chunks}) != len(chunks):
                raise ValueError(f"IDs duplicados en {name}")
            if any(c["text"] != text[c["start"]:c["end"]] for c in chunks):
                raise ValueError(f"Los offsets no reconstruyen los chunks de {name}")
            metrics.append(metric)
            evidence.extend({"experiment": name, **row} for row in evidence_preservation(
                chunks, text, references.get("evidence_cases", []), located
            ))
        with artifact_stage(runs, prefix=".processing-") as directory:
            stage = Path(directory).resolve()
            if not stage.is_relative_to((root / "data/processed").resolve()):
                raise ValueError("El staging de resultados sale del proyecto.")
            for name, records in {
                "pages.jsonl": pages, "blocks.jsonl": blocks, "located_blocks.jsonl": located,
                "unresolved_objects.jsonl": objects, "page_audit.jsonl": audit,
            }.items():
                write_jsonl(stage / name, records)
            for name, chunks in experiments.items():
                write_jsonl(stage / f"chunks_{name}.jsonl", chunks)
            (stage / "corpus.txt").write_text(text, encoding="utf-8")
            write_csv(stage / "metrics.csv", metrics)
            write_csv(stage / "evidence_preservation.csv", evidence)
            write_csv(stage / "parsing_reference_comparison.csv", comparison)
            artifacts = {path.name: sha256_file(path) for path in sorted(stage.iterdir())}
            report = {
                **inputs, "run_id": run_id, "status": "complete",
                "created_at": datetime.now(UTC).isoformat(), "pages": len(pages),
                "blocks_total": len(blocks), "retrieval_blocks": len(located),
                "selected_characters": len(text), "unresolved_objects": len(objects),
                "decorative_images_excluded": sum(len(p["decorative_images"]) for p in audit),
                "corpus_fingerprint": fingerprint([(b["block_id"], b["start"], b["end"])
                                                   for b in located]),
                "metrics": metrics, "seconds": time.perf_counter() - started,
                "artifact_sha256": artifacts, "encoder_executed": False, "llm_calls": 0,
                "review_status": references["status"],
            }
            write_json(stage / "manifest.json", report)
            stage.rename(destination)
        _update_latest(root, output, destination, run_id)
        return report
    finally:
        lock_stream.close()
        lock.unlink(missing_ok=True)


def _update_latest(root, output, destination, run_id):
    temporary = output / ".latest.json.tmp"
    write_json(temporary, {"run_id": run_id, "path": destination.relative_to(root).as_posix()})
    temporary.replace(output / "latest.json")
