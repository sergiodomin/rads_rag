"""Prueba pequeña del reranker local sobre dos pasajes sintéticos, sin LLM."""

import json
import os
from pathlib import Path

import httpx

from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.retrieval.reranking import LocalReranker


def main():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = Path(__file__).resolve().parents[1]
    try:
        response = httpx.get("http://127.0.0.1:11434/api/ps", timeout=5)
        response.raise_for_status()
        if response.json().get("models"):
            raise ValueError("Espera a que Ollama descargue el generador de memoria.")
    except httpx.ConnectError:
        pass  # El reranker no necesita que Ollama esté arrancado.
    sources = [
        {"text": "Paris is the capital of France."},
        {"text": "Jupiter is a planet in the solar system."},
    ]
    scores, pairs, metrics = LocalReranker(root, "qwen3-reranker-0.6b").score(
        "¿Cuál es la capital de Francia?", sources
    )
    report = {
        "scores": scores,
        "pairs": pairs,
        "metrics": metrics,
        "relevant_above_distractor": scores[0] > scores[1],
    }
    atomic_json(root / "output/reranker_smoke.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not report["relevant_above_distractor"]:
        raise ValueError("La prueba sintética no ordenó el pasaje relevante primero.")


if __name__ == "__main__":
    main()
