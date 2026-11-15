"""Procesa la revisión abierta BI-RADS y publica su índice estructural local."""
import gc
import json
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np

from radiology_rag.artifacts import artifact_stage
from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.chunking import TokenCounter
from radiology_rag.ingestion.download import validate_pdf
from radiology_rag.ingestion.jats import build_passages
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import project_root, within
from radiology_rag.retrieval.dense import encode_cached, load_local_encoder


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(within(folder, name)) != digest:
            raise ValueError("Artefacto modificado: " + name)


def main():
    import psutil
    import torch

    root = project_root()
    catalog_path = root / "data/catalog/documents.v1.json"
    catalog = load_catalog(catalog_path)
    doc = next(d for d in catalog["documents"] if d["id"] == "birads-v2025-review-2026")
    pdf = within(root, doc["local_path"])
    identity = validate_pdf(pdf, doc)
    provenance = json.loads(pdf.with_suffix(".source.json").read_text(encoding="utf-8"))
    if provenance["sha256"] != identity["sha256"] or provenance["document_id"] != doc["id"]:
        raise ValueError("Procedencia del PDF no compatible.")
    xml = pdf.with_suffix(".xml")
    if sha256_file(xml) != doc["repository_xml_sha256"]:
        raise ValueError("El XML no corresponde a la fuente fijada.")
    counter = TokenCounter.from_project(root)
    passages, objects = build_passages(xml.read_bytes(), doc, counter)
    if sum(o["kind"] == "table" for o in objects) != 6:
        raise ValueError("Se esperaban seis tablas en esta revisión.")
    models = json.loads((root / "data/catalog/embedding_models.v1.json").read_text())["models"]
    inputs = {"document_id": doc["id"], "source_sha256": identity["sha256"],
              "xml_sha256": sha256_file(xml), "models": models,
              "collections": ["structural_512"], "tokenizer_sha256": counter.spec["sha256"],
              "implementation": {p: sha256_file(root / p) for p in (
                  "scripts/prepare_birads_index.py", "src/radiology_rag/ingestion/jats.py",
                  "src/radiology_rag/retrieval/dense.py")},
              "environment": {p: version(p) for p in ("torch", "transformers", "numpy")}}
    run_id = fingerprint(inputs)[:16]
    processed = root / f"data/processed/{doc['id']}/jats/runs"
    indices = root / f"data/indices/{doc['id']}/03_dense/runs"
    reports = root / f"data/runs/{doc['id']}/03_embeddings"
    for folder in (processed, indices, reports / "runs"):
        folder.mkdir(parents=True, exist_ok=True)
    with (reports / ".publish.lock").open("x", encoding="utf-8") as lock:
        try:
            output = processed / run_id
            if not output.exists():
                with artifact_stage(processed, prefix=".jats-") as stage:
                    write(stage / "objects.json", objects)
                    (stage / "passages.jsonl").write_text("".join(
                        json.dumps(p, ensure_ascii=False) + "\n" for p in passages
                    ), encoding="utf-8")
                    write(stage / "manifest.json", {"inputs": inputs, "run_id": run_id,
                          "artifact_sha256": {p.name: sha256_file(p) for p in stage.iterdir()}})
                    stage.rename(output)
            verify(output)
            destination = indices / run_id
            costs = {}
            if not destination.exists():
                if not torch.cuda.is_available() or psutil.virtual_memory().available < 3 * 1024**3:
                    raise ValueError("Se requiere CUDA y al menos 3 GiB de RAM libre.")
                torch.set_num_threads(2)
                with artifact_stage(indices, prefix=".index-") as stage:
                    for spec in models:
                        print("Codificando", spec["id"], len(passages), "pasajes", flush=True)
                        encoder = load_local_encoder(root, spec, device="cuda:0", dtype="float16")
                        try:
                            matrix, costs[spec["id"]] = encode_cached(root, encoder, spec, [
                                {"id": p["chunk_id"], "text": p.get("embedding_text", p["text"])}
                                for p in passages
                            ], "passage", batch_size=1)
                        finally:
                            del encoder
                            gc.collect()
                            torch.cuda.empty_cache()
                        folder = stage / spec["id"] / "structural_512"
                        folder.mkdir(parents=True)
                        np.save(folder / "embeddings.npy", matrix, allow_pickle=False)
                        (folder / "passages.jsonl").write_bytes(
                            (output / "passages.jsonl").read_bytes())
                    write(stage / "manifest.json", {"run_id": run_id, "inputs": inputs,
                          "artifact_sha256": {p.relative_to(stage).as_posix(): sha256_file(p)
                                              for p in stage.rglob("*") if p.is_file()}})
                    stage.rename(destination)
            verify(destination)
            report = reports / "runs" / run_id
            if not report.exists():
                with artifact_stage(reports / "runs", prefix=".report-") as stage:
                    write(stage / "build.json", {"costs": costs, "passages": len(passages),
                          "tables": 6, "figures_interpreted": 0, "evaluation_executed": False})
                    write(stage / "manifest.json", {"run_id": run_id, "status": "complete",
                          "inputs": inputs, "created_at": datetime.now(UTC).isoformat(),
                          "index_manifest_sha256": sha256_file(destination / "manifest.json"),
                          "artifact_sha256": {"build.json": sha256_file(stage / "build.json")}})
                    stage.rename(report)
            verify(report)
            temporary = reports / ".latest.json.tmp"
            write(temporary, {"run_id": run_id, "path": report.relative_to(root).as_posix(),
                              "index_path": destination.relative_to(root).as_posix()})
            temporary.replace(reports / "latest.json")
        finally:
            lock.close()
            (reports / ".publish.lock").unlink(missing_ok=True)
    # Enable only after all immutable artifacts are complete and verified.
    catalog["chat_document_ids"] = [catalog["active_document_id"], doc["id"]]
    write(catalog_path, catalog)
    print("BI-RADS preparado:", run_id, len(passages), "pasajes; sin evaluación clínica.")


if __name__ == "__main__":
    main()
