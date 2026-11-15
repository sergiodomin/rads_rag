"""Regresión de recuperación real; no necesita CUDA, LLM ni credenciales."""
from pathlib import Path

import numpy as np
import pytest

from radiology_rag.chat import ChatEngine
from radiology_rag.config import Settings


@pytest.mark.parametrize("question,category", [
    ("qué tengo que tener en cuenta para identificar un birdas 3", "3"),
    ("¿Qué criterios definen BI-RADS 3?", "3"),
    ("What criteria define BI-RADS category 3?", "3"),
    ("¿Qué cambio describe BI-RADS v2025 para la categoría 6?", "6"),
])
def test_real_table_category_beats_a_deliberately_wrong_dense_neighbor(question, category):
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/runs/birads-v2025-review-2026/03_embeddings/latest.json").exists():
        pytest.skip("Corpus local no preparado; la regresión sintética siempre se ejecuta.")
    engine = ChatEngine(root, Settings())
    rows, matrix = engine.collections["qwen3-0.6b", "structural_512"]
    distractor = next(i for i, r in enumerate(rows)
                      if r.get("xml_element_id", "").endswith("t001"))

    class DistractedEncoder:
        def tokenizer(self, text):
            return {"input_ids": list(range(len(text.split())))}

        def encode(self, _texts):
            return np.array([matrix[distractor]])

    engine._encoder_id, engine._encoder = "qwen3-0.6b", DistractedEncoder()
    result = engine.respond({"message": question, "system": "all",
                             "reranker": "none",
                             "encoder": "qwen3-0.6b", "collection": "structural_512"},
                            search_only=True)
    first = result["sources"][0]
    assert first["system"] == "BI-RADS" and first["assessment_categories"] == [category]
    assert first["exact_category_match"]
    assert first["xml_element_id"].endswith("t005")
    assert "2013 Definition" in first["table_headers"]
    assert "v2025 Update" in first["table_headers"] and "Notes:" in first["text"]
    assert result["metrics"]["requested_categories"] == {"BI-RADS": [category]}
