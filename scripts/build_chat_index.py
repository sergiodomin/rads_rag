"""Construye índices del chat sin requerir ni inventar anotaciones de evaluación."""
import gc
import json
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import torch

from radiology_rag.artifacts import artifact_stage
from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.chunking import TokenCounter
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import project_root, within
from radiology_rag.retrieval.dense import (
    encode_cached,
    load_local_encoder,
    prepare_passages,
    read_jsonl,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_artifacts(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(within(folder, name)) != digest:
            raise ValueError("Un artefacto del índice ha cambiado.")
    return manifest


def main():
    root = project_root()
    catalog = load_catalog(root / "data/catalog/documents.v1.json")
    document = catalog["active_document_id"]
    if document != "pirads-2.1-2019":
        raise ValueError("El chat actual está preparado para PI-RADS v2.1; ver roadmap multi-RADS.")
    pointer = json.loads((root / f"data/processed/{document}/02_chunking/latest.json").read_text())
    source = within(root, pointer["path"])
    source_manifest = check_artifacts(source)
    text = (source / "corpus.txt").read_text(encoding="utf-8")
    blocks = read_jsonl(source / "located_blocks.jsonl")
    counter = TokenCounter.from_project(root)
    collections = {name: prepare_passages(read_jsonl(source / f"chunks_{name}.jsonl"),
                                         text, blocks, counter)
                   for name in ("fixed_512_15", "structural_512")}
    passages = {p["chunk_id"]: p for rows in collections.values() for p in rows}
    records = [{"id": p["chunk_id"], "text": p["text"]} for p in passages.values()]
    positions = {p["id"]: i for i, p in enumerate(records)}
    models = json.loads((root / "data/catalog/embedding_models.v1.json").read_text())["models"]
    inputs = {"kind": "standalone_chat_index", "source_run_id": source_manifest["run_id"],
              "source_sha256": source_manifest["source_sha256"],
              "source_manifest_sha256": sha256_file(source / "manifest.json"),
              "models": models, "collections": list(collections), "device": "cuda:0",
              "dtype": "float16", "batch_size": 1,
              "environment": {p: version(p) for p in ("torch", "transformers", "numpy")},
              "builder_sha256": sha256_file(root / "scripts/build_chat_index.py"),
              "encoder_sha256": sha256_file(root / "src/radiology_rag/retrieval/dense.py")}
    run_id = fingerprint(inputs)[:16]
    indices = root / f"data/indices/{document}/03_dense/runs"
    reports = root / f"data/runs/{document}/03_embeddings"
    indices.mkdir(parents=True, exist_ok=True)
    (reports / "runs").mkdir(parents=True, exist_ok=True)
    destination, report_out = indices / run_id, reports / "runs" / run_id
    lock = reports / ".publish.lock"
    with lock.open("x", encoding="utf-8") as stream:
        try:
            costs = {}
            if destination.exists():
                check_artifacts(destination)
            else:
                if not torch.cuda.is_available():
                    raise ValueError("El índice actual requiere el entorno CUDA del laboratorio.")
                torch.set_num_threads(2)
                with artifact_stage(indices, prefix=".index-") as stage:
                    for spec in models:
                        encoder = load_local_encoder(root, spec, device="cuda:0", dtype="float16")
                        try:
                            matrix, costs[spec["id"]] = encode_cached(
                                root, encoder, spec, records, "passage", batch_size=1
                            )
                        finally:
                            del encoder
                            gc.collect()
                            torch.cuda.empty_cache()
                        for name, rows in collections.items():
                            folder = stage / spec["id"] / name
                            folder.mkdir(parents=True)
                            np.save(folder / "embeddings.npy", matrix[
                                [positions[p["chunk_id"]] for p in rows]], allow_pickle=False)
                            (folder / "passages.jsonl").write_text("".join(
                                json.dumps(p, ensure_ascii=False) + "\n" for p in rows
                            ), encoding="utf-8")
                    artifacts = {p.relative_to(stage).as_posix(): sha256_file(p)
                                 for p in sorted(stage.rglob("*")) if p.is_file()}
                    write_json(stage / "manifest.json", {"run_id": run_id, "inputs": inputs,
                               "artifact_sha256": artifacts, "normalized": True})
                    stage.rename(destination)
            if report_out.exists():
                check_artifacts(report_out)
            else:
                with artifact_stage(reports / "runs", prefix=".report-") as stage:
                    write_json(stage / "build.json", {"costs": costs, "evaluation_executed": False})
                    write_json(stage / "manifest.json", {
                        "run_id": run_id, "status": "complete", "inputs": inputs,
                        "created_at": datetime.now(UTC).isoformat(),
                        "index_path": destination.relative_to(root).as_posix(),
                        "index_manifest_sha256": sha256_file(destination / "manifest.json"),
                        "artifact_sha256": {"build.json": sha256_file(stage / "build.json")},
                        "n_cases": 0, "n_queries": 0, "evaluation_executed": False,
                        "encoder_executed": True, "llm_calls": 0,
                        "review_status": "local_annotations_not_required_for_index",
                    })
                    stage.rename(report_out)
            temporary = reports / ".latest.json.tmp"
            write_json(temporary, {"run_id": run_id,
                                  "path": report_out.relative_to(root).as_posix(),
                                  "index_path": destination.relative_to(root).as_posix()})
            temporary.replace(reports / "latest.json")
        finally:
            stream.close()
            lock.unlink(missing_ok=True)
    print("Índice del chat preparado; no se ha ejecutado evaluación:", run_id)


if __name__ == "__main__":
    main()
