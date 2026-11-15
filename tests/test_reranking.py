"""Orden, selección y presupuestos sin descargar ni cargar modelos."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import radiology_rag.chat as chat
from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.retrieval.reranking import (
    PREFIX,
    SUFFIX,
    model_spec,
    order_by_scores,
    pair_tokens,
    select_context,
)
from scripts.compare_reranking import compare_runs


class CharacterTokenizer:
    def encode(self, text, **_):
        return [ord(c) for c in text]


def source(identifier, system="PI-RADS", parent=None, exact=False):
    return {
        "id": "C1",
        "chunk_id": identifier,
        "system": system,
        "document_id": system,
        "parent_chunk_id": parent or identifier,
        "exact_category_match": exact,
    }


def test_pair_budget_preserves_query_template_and_records_document_crop():
    tokenizer = CharacterTokenizer()
    instruction, query = "Retrieve", "question with number 6 and negation NOT"
    header = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "
    limit = len(PREFIX) + len(header) + len(SUFFIX) + 4
    ids, info = pair_tokens(tokenizer, instruction, query, "DOCUMENT", limit)
    text = "".join(chr(i) for i in ids)
    assert text == PREFIX + header + "DOCU" + SUFFIX
    assert info["document_truncated"] and info["scored_document_tokens"] == 4
    assert info["original_document_tokens"] == 8 and info["pair_tokens"] == limit
    with pytest.raises(ValueError, match="presupuesto"):
        pair_tokens(tokenizer, instruction, query, "document", 10)


def test_scores_are_stable_and_invalid_results_fail_instead_of_silent_fallback():
    sources = [source("a"), source("b"), source("c")]
    assert order_by_scores(sources, [0.4, 0.9, 0.9]) == [1, 2, 0]
    for values in ([1], [0.2, float("nan"), 0.8]):
        with pytest.raises(ValueError, match="Puntuaciones"):
            order_by_scores(sources, values)


def test_selection_deduplicates_parents_and_reassigns_final_citation_ids():
    sources = [source("a", parent="same"), source("a-child", parent="same"), source("b")]
    selected = select_context(sources)
    assert [s["chunk_id"] for s in selected] == ["a", "b"]
    assert [s["id"] for s in selected] == ["S1", "S2"]


def test_comparison_keeps_two_places_per_system_and_protected_scope():
    sources = [source(f"b{i}", "BI-RADS") for i in range(5)] + [source(f"p{i}") for i in range(3)]
    result = select_context(sources, comparison=True, protected_ids=["p2", "b4"])
    assert {s["chunk_id"] for s in result} == {"p2", "b4", "b0", "p0"}
    assert len([s for s in result if s["system"] == "PI-RADS"]) == 2
    assert (
        select_context([source("wrong"), source("target", exact=True)], category_first=True)[0][
            "chunk_id"
        ]
        == "target"
    )


@pytest.mark.parametrize("policy,expected", [("parent", "full parent"), ("chunk", "topic child")])
def test_pipeline_reranks_before_citations_with_full_candidate_trace(
    tmp_path, monkeypatch, policy, expected
):
    engine = object.__new__(chat.ChatEngine)
    engine.root, engine.run_id = tmp_path, "unit"
    engine.document = {"id": "unit", "system": "PI-RADS"}
    engine.specs = {"unit": {"query_prefix": "", "encoder_max_tokens": 512}}
    engine._encoder, engine._encoder_id = None, None
    rows = [
        {
            "chunk_id": f"c{i}",
            "parent_chunk_id": f"p{i}",
            "text": "topic child",
            "parent_text": "full parent",
            "pages": [i + 1],
            "source_sha256": "unit-digest",
            "spans": [{"page_label": str(i + 1)}],
        }
        for i in range(2)
    ]
    engine.collections = {("unit", "structural_512"): (rows, np.eye(2))}

    class Scorer:
        def __init__(self, root, identifier):
            assert root == tmp_path and identifier == "synthetic-reranker"

        def score(self, query, candidates):
            assert engine._encoder is None and engine._phase == "reranking"
            assert all(c["text"] == expected for c in candidates)
            assert query == "topic"
            return [0.1, 0.9], [{}, {}], {"pairs": 2}

    monkeypatch.setattr(chat, "LocalReranker", Scorer)
    sources, metrics = engine._retrieve(
        "topic",
        [],
        "unit",
        "structural_512",
        method="bm25",
        include_trace=True,
        candidate_count=20,
        reranker="synthetic-reranker",
        context_policy=policy,
    )
    assert sources[0]["chunk_id"] == "c1" and sources[0]["id"] == "S1"
    assert sources[0]["source_sha256"] == "unit-digest"
    assert sources[0]["score"] is None  # A reranker score is not a cosine score.
    assert metrics["reranking"]["pairs"] == 2
    trace = metrics["retrieval_trace"]
    assert trace["baseline_sources"][0]["chunk_id"] == "c0"
    assert trace["candidates"][0]["rerank_rank"] == 2
    assert trace["candidates"][1]["selected_rank"] == 1


def test_unprepared_model_rejected_without_network_access(tmp_path):
    with pytest.raises(ValueError, match="desconocido"):
        model_spec(tmp_path, "unknown", verify=True)


def test_same_size_tampered_weights_fail_hash_verification(tmp_path):
    import hashlib

    (tmp_path / "weights.bin").write_bytes(b"model")
    atomic_json(
        tmp_path / "data/catalog/reranker_models.v1.json",
        {
            "schema_version": 1,
            "models": [
                {
                    "id": "unit",
                    "local_path": ".",
                    "files": [
                        {
                            "path": "weights.bin",
                            "bytes": 5,
                            "sha256": hashlib.sha256(b"model").hexdigest(),
                        }
                    ],
                }
            ],
        },
    )
    assert model_spec(tmp_path, "unit", verify=True)
    (tmp_path / "weights.bin").write_bytes(b"other")
    with pytest.raises(ValueError, match="cambiado"):
        model_spec(tmp_path, "unit", verify=True)


def test_comparison_rejects_candidate_changes_and_keeps_paired_metrics():
    record = {
        "case_id": "unit",
        "language": "es",
        "reference_status": "draft",
        "automatic": {
            "retrieval": {"evidence_recall": 0.5},
            "retrieval_candidate_pool": {"evidence_recall": 1},
        },
        "trace": {
            "retrieval": {
                "candidate_sources": [{"document_id": "unit", "chunk_id": "c", "text": "synthetic"}]
            }
        },
        "elapsed_seconds": 1,
        "metrics": {"reranking": {}},
    }
    baseline = {
        "snapshot": {
            "dataset_sha256": "unit",
            "index_runs": "fixed",
            "config": {"reranker": "none", "candidate_count": 20},
        },
        "manifest": {"run_id": "a"},
        "records": [record],
    }
    reranked = deepcopy(baseline)
    reranked["snapshot"]["config"]["reranker"] = "unit-reranker"
    reranked["records"][0]["automatic"]["retrieval"]["evidence_recall"] = 1
    assert compare_runs(baseline, reranked)["summary"]["reranked_mean_recall"] == 1
    reranked["records"][0]["trace"]["retrieval"]["candidate_sources"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="candidatos"):
        compare_runs(baseline, reranked)
    reranked["snapshot"]["config"]["candidate_count"] = 40
    with pytest.raises(ValueError, match="sólo el reranker"):
        compare_runs(baseline, reranked)


def test_encoder_local_reference_is_released_before_scorer(tmp_path, monkeypatch):
    engine = object.__new__(chat.ChatEngine)
    engine.root, engine.run_id, engine.document = tmp_path, "unit", {"id": "unit"}
    engine.specs = {"unit": {"query_prefix": "", "encoder_max_tokens": 512}}
    released = []

    class Encoder:
        def tokenizer(self, _):
            return {"input_ids": [1]}

        def encode(self, _):
            return np.array([[1, 0]])

        def __del__(self):
            released.append(True)

    def load(_):
        engine._encoder = Encoder()
        return engine._encoder

    engine._load_encoder = load
    engine.release_encoder = lambda: setattr(engine, "_encoder", None)
    row = {
        "chunk_id": "c",
        "parent_chunk_id": "p",
        "text": "topic",
        "pages": [1],
        "spans": [{"page_label": "1"}],
    }
    engine.collections = {("unit", "structural_512"): ([row], np.array([[1, 0]]))}

    def score(*_):
        assert released, "A local encoder variable must not keep GPU weights alive"
        return [0.9], [{}], {}

    monkeypatch.setattr(chat, "LocalReranker", lambda *_: SimpleNamespace(score=score))
    engine._retrieve("topic", [], "unit", "structural_512", reranker="unit")
