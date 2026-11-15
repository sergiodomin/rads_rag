"""Comparación pareada en desarrollo usando la API local del laboratorio."""

import argparse
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.ingestion.parsing import fingerprint

CASES = [
    f"{group}-{language}"
    for group in ("pi-patient-information", "bi-category-3", "conversation-topic-switch")
    for language in ("es", "en")
]


def execute(client, config):
    response = client.post("/api/lab/start", json=config)
    response.raise_for_status()
    run_id = response.json()["run_id"]
    print(f"Run: {run_id} ({config['reranker']})", flush=True)
    while True:
        state = client.get("/api/lab/state").raise_for_status().json()
        if state["run_id"] != run_id:
            raise ValueError("Ha cambiado el lote activo; revisa la ejecución en /lab.")
        if state["status"] != "running":
            break
        time.sleep(2)
    run = client.get(f"/api/lab/runs/{run_id}").raise_for_status().json()
    if state["status"] != "complete":
        raise ValueError("Lote incompleto; inspecciona sus errores en /lab.")
    return run


def compare_runs(before, after):
    a, b = before["snapshot"], after["snapshot"]
    if a["dataset_sha256"] != b["dataset_sha256"] or a["index_runs"] != b["index_runs"]:
        raise ValueError("No se pueden emparejar datasets o índices diferentes.")
    if a.get("code_sha256") != b.get("code_sha256") or a.get("encoder_profiles") != b.get(
        "encoder_profiles"
    ):
        raise ValueError("No se pueden emparejar código o revisiones de encoder diferentes.")
    comparable = {k: v for k, v in a["config"].items() if k != "reranker"}
    if comparable != {k: v for k, v in b["config"].items() if k != "reranker"}:
        raise ValueError("La comparación debe cambiar sólo el reranker.")
    right = {r["case_id"]: r for r in after["records"]}
    pairs = []
    for left in before["records"]:
        candidate = right[left["case_id"]]

        def signature(record):
            return [
                (s["document_id"], s["chunk_id"], fingerprint(s["text"]))
                for s in record["trace"]["retrieval"]["candidate_sources"]
            ]

        if signature(left) != signature(candidate):
            raise ValueError("Los candidatos no son idénticos: " + left["case_id"])
        pairs.append(
            {
                "case_id": left["case_id"],
                "language": left["language"],
                "reference_status": left["reference_status"],
                "candidate_pool_recall": candidate["automatic"]["retrieval_candidate_pool"][
                    "evidence_recall"
                ],
                "baseline_recall": left["automatic"]["retrieval"]["evidence_recall"],
                "reranked_recall": candidate["automatic"]["retrieval"]["evidence_recall"],
                "baseline_seconds": left["elapsed_seconds"],
                "reranked_seconds": candidate["elapsed_seconds"],
                "reranking_metrics": candidate["metrics"]["reranking"],
            }
        )
    valid = [
        p for p in pairs if p["baseline_recall"] is not None and p["reranked_recall"] is not None
    ]
    return {
        "baseline_run": before["manifest"]["run_id"],
        "reranked_run": after["manifest"]["run_id"],
        "dataset_sha256": a["dataset_sha256"],
        "config": comparable,
        "candidates_verified_identical": True,
        "cases": pairs,
        "summary": {
            "cases": len(pairs),
            "positive_cases": len(valid),
            "baseline_mean_recall": sum(p["baseline_recall"] for p in valid) / len(valid)
            if valid
            else None,
            "reranked_mean_recall": sum(p["reranked_recall"] for p in valid) / len(valid)
            if valid
            else None,
        },
        "limitations": [
            "Development subset with draft references; no clinical validation.",
            "No performance benchmark: OS caches not controlled.",
            "No generation calls; does not evaluate answer quality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--candidate-count", choices=[20, 40], type=int, default=20)
    args = parser.parse_args()
    url = urlsplit(args.url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        parser.error("Sólo se admite la API local HTTP loopback.")
    with httpx.Client(base_url=args.url, timeout=30) as client:
        data = client.get("/api/lab").raise_for_status().json()["dataset"]
        cases = {c["id"]: c for c in data["cases"]}
        if any(cases[c]["split"] != "dev" for c in CASES):
            raise ValueError("Esta muestra debe permanecer en desarrollo.")
        config = {
            "case_ids": CASES,
            "conditions": ["retrieval_only"],
            "models": [],
            "encoder": "qwen3-0.6b",
            "retriever": "hybrid",
            "collection": "structural_512",
            "context_policy": "parent",
            "candidate_count": args.candidate_count,
            "reranker": "none",
        }
        baseline = execute(client, config)
        reranked = execute(client, {**config, "reranker": "qwen3-reranker-0.6b"})
    report = compare_runs(baseline, reranked)
    path = Path(__file__).resolve().parents[1] / "output/lab/reranking_comparison.json"
    atomic_json(path, report)
    print(report["summary"], flush=True)
    print("Informe: output/lab/reranking_comparison.json", flush=True)


if __name__ == "__main__":
    main()
