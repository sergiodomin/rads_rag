"""Controles experimentales, procedencia y concurrencia sin pesos ni GPU."""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import radiology_rag.evaluation.lab as module
from radiology_rag.chat import BusyError, ChatEngine
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab import EvaluationLab, retrieval_assessment, summarize
from radiology_rag.evaluation.lab_dataset import (
    DATASET,
    atomic_json,
    evidence_matches,
    load_dataset,
    resolve_seed,
    validate_dataset,
)
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.web.server import build_server


@pytest.fixture
def lab(tmp_path):
    (tmp_path / "manual.pdf").write_bytes(b"synthetic source")
    source = {
        "id": "S1",
        "document_id": "synthetic",
        "system": "PI-RADS",
        "source_sha256": sha256_file(tmp_path / "manual.pdf"),
        "pages": [1],
        "printed_pages": ["1"],
        "chunk_id": "reference",
        "text": "synthetic reference fact",
        "sections": [],
        "document": "Synthetic",
        "document_role": "test",
        "document_version": "1",
    }
    case = {
        "id": "unit-es",
        "group_id": "unit",
        "split": "dev",
        "language": "es",
        "question": "What is the synthetic fact?",
        "system": "PI-RADS",
        "history": [],
        "answerable": True,
        "type": "direct",
        "review_status": "draft",
        "review_notes": "",
        "reference_answer": "synthetic reference fact",
        "evidence": [
            {
                "id": "e1",
                "source": source,
                "selector": {"page_number": 1, "anchor": "reference fact"},
            }
        ],
    }
    dataset = {"schema_version": 1, "revision": 1, "cases": [case]}
    atomic_json(tmp_path / DATASET, dataset)
    # Snapshot code fingerprints must resolve; fixtures contain no clinical text.
    project = Path(__file__).resolve().parents[1]
    for name in [
        "evaluation/lab.py",
        "evaluation/lab_dataset.py",
        "chat.py",
        "retrieval/hybrid.py",
        "retrieval/reranking.py",
        "generation/scope.py",
        "generation/terminology.py",
        "generation/context.py",
        "generation/ollama.py",
        "ingestion/jats.py",
    ]:
        dest = tmp_path / "src/radiology_rag" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((project / "src/radiology_rag" / name).read_bytes())
    atomic_json(tmp_path / "data/catalog/system_profiles.v1.json", {"schema_version": 1})
    atomic_json(
        tmp_path / "data/catalog/lab_generators.v1.json",
        {
            "schema_version": 1,
            "generators": [{"profile": "data/catalog/unit.json", "budget_format": "qwen3_chatml"}],
        },
    )
    atomic_json(
        tmp_path / "data/catalog/unit.json", {"model": "unit-model", "ollama_digest": "unit-digest"}
    )
    doc = {
        "id": "synthetic",
        "local_path": "manual.pdf",
        "system": "PI-RADS",
        "expected_sha256": source["source_sha256"],
    }
    engine = SimpleNamespace(
        root=tmp_path,
        _lock=threading.Lock(),
        _phase="ready",
        document=doc,
        documents={"synthetic": doc},
        specs={"e5-base": {}},
        run_id="synthetic-index",
        document_collections={"synthetic": ["structural_512"]},
        settings=Settings(base_url="http://127.0.0.1:11434", model="unit-model", num_predict=128),
        release_encoder=lambda: None,
        _scope_answer=lambda *a: None,
    )
    actual = {**source, "chunk_id": "retrieved", "text": "synthetic retrieved fact"}
    engine._retrieve = lambda *a, **kw: ([actual], {"retrieval_trace": {"candidates": []}})
    return EvaluationLab(engine)


def config(**changes):
    return {
        "case_ids": ["unit-es"],
        "conditions": ["retrieval_only"],
        "models": [],
        "retriever": "hybrid",
        "encoder": "e5-base",
        "collection": "structural_512",
        **changes,
    }


def test_evidence_coverage_uses_identity_anchor_and_no_duplicate_credit(lab):
    case = load_dataset(lab.root)["cases"][0]
    source = case["evidence"][0]["source"]
    metrics = retrieval_assessment(case, [source, source])
    assert metrics["evidence_recall"] == 1 and metrics["mrr"] == 1
    assert metrics["reference_status"] == "draft"
    assert not evidence_matches({**source, "source_sha256": "changed"}, case["evidence"][0])
    item = deepcopy(case["evidence"][0])
    item["source"]["xml_sha256"] = "expected"
    assert not evidence_matches(source, item)
    assert retrieval_assessment({**case, "evidence": []}, [])["evidence_recall"] is None


def test_dataset_keeps_translations_in_same_partition(lab):
    data = load_dataset(lab.root)
    other = {**data["cases"][0], "id": "unit-en", "split": "test"}
    data["cases"].append(other)
    with pytest.raises(ValueError, match="cruza"):
        validate_dataset(data)


def test_public_seed_offsets_require_exact_local_text():
    blocks = [{"block_id": "synthetic", "page_number": 2, "text": "local evidence"}]
    item = {
        "block_id": "synthetic",
        "page_number": 2,
        "start": 6,
        "end": 14,
        "block_text_sha256": fingerprint("local evidence"),
    }
    assert resolve_seed(blocks, item)[1]["anchor"] == "evidence"
    blocks[0]["text"] = "changed evidence"
    with pytest.raises(ValueError, match="offsets"):
        resolve_seed(blocks, item)


def test_generation_call_limit_counts_all_conditions(lab):
    dataset = load_dataset(lab.root)
    for i in range(1, 5):
        dataset["cases"].append({**dataset["cases"][0], "id": f"unit-{i}"})
    with pytest.raises(ValueError, match="12 llamadas"):
        lab._validate_config(
            config(
                case_ids=[c["id"] for c in dataset["cases"]],
                conditions=["no_context", "reference_context", "retrieved_context"],
                models=["unit-model"],
            ),
            dataset,
        )


@pytest.mark.parametrize("condition", ["no_context", "reference_context", "retrieved_context"])
def test_controls_send_correct_context_and_keep_raw_output(lab, monkeypatch, condition):
    requests = []

    class Client:
        def __init__(self, settings):
            assert settings.is_local and settings.max_retries == 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def chat(self, messages):
            requests.append(messages)
            return SimpleNamespace(
                content="An answer with invalid ID [S99]",
                done_reason="stop",
                metrics=lambda: {"eval_count": 12},
            )

    monkeypatch.setattr(module, "OllamaCloudClient", Client)
    monkeypatch.setattr(lab, "_budget", lambda *_: {"input_tokens": 42})
    case = load_dataset(lab.root)["cases"][0]
    result = lab._evaluate(
        case,
        condition,
        "unit-model",
        lab._validate_config(config(), load_dataset(lab.root)),
        lab.generator_profiles(),
    )
    assert result["status"] == "complete" and result["generation_executed"]
    assert result["raw_answer"].endswith("[S99]")
    assert result["automatic"]["citation_semantic_support"] is None
    if condition == "no_context":
        assert result["context_sources"] == [] and result["answer"] == result["raw_answer"]
        assert result["automatic"]["citations_without_context"]
    else:
        context = json.loads(requests[0][-1]["content"])["document_fragments"]
        expected = "reference fact" if condition == "reference_context" else "retrieved fact"
        assert expected in context[0]["text"]
        assert result["abstained"] and result["answer"] != result["raw_answer"]
        assert not result["automatic"]["citation_ids_valid"]
    assert result["elapsed_seconds"] >= 0


def test_product_fields_are_distinct_from_free_generation(lab, monkeypatch):
    lab.engine._scope_answer = lambda *_: {
        "answer": "verified fields [S1]",
        "abstained": False,
        "abstention_reason": None,
        "cited_sources": ["S1"],
        "truncated": False,
        "answer_mode": "verified_scope_fields",
        "metrics": {"llm_calls": 0},
    }
    monkeypatch.setattr(lab, "_budget", lambda *_: pytest.fail("Must not load tokenizer"))
    case = load_dataset(lab.root)["cases"][0]
    result = lab._evaluate(case, "product", "unit-model", config(), lab.generator_profiles())
    assert result["answer_mode"] == "verified_scope_fields"
    assert not result["generation_executed"]


def test_cancel_serializes_with_chat_and_preserves_snapshot(lab):
    entered, release = threading.Event(), threading.Event()
    original = lab.engine._retrieve

    def retrieve(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    lab.engine._retrieve = retrieve
    state = lab.start(config())
    assert entered.wait(5)
    assert lab.engine._lock.locked()
    with pytest.raises(BusyError):
        lab.start(config())
    with pytest.raises(BusyError):
        lab.review_case({"case_id": "unit-es", "reference_answer": "changed"})
    lab.cancel()
    release.set()
    lab._thread.join(5)
    assert not lab._thread.is_alive() and not lab.engine._lock.locked()
    assert lab.state()["status"] == "cancelled" and lab.engine._phase == "ready"
    lab.review_case({"case_id": "unit-es", "reference_answer": "human edited", "reviewed": True})
    run = lab.read_run(state["run_id"])
    assert run["snapshot"]["dataset"]["revision"] == 1
    assert load_dataset(lab.root)["revision"] == 2
    assert run["records"][0]["reference_status"] == "draft"
    assert summarize(run["records"])[0]["semantic_accuracy"] is None


def test_failed_record_releases_lock_and_other_runs_can_start(lab):
    def fail(*_, **__):
        raise ValueError("synthetic failure")

    lab.engine._retrieve = fail
    state = lab.start(config())
    lab._thread.join(5)
    assert lab.state()["status"] == "completed_with_errors"
    assert lab.read_run(state["run_id"])["records"][0]["error"] == "synthetic failure"
    assert not lab.engine._lock.locked()


def test_changed_source_rejected_without_leaking_lock(lab):
    (lab.root / "manual.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="documento"):
        lab.start(config())
    assert not lab.engine._lock.locked()


def test_uninstalled_generator_and_cloud_rejected(lab, monkeypatch):
    monkeypatch.setattr(lab, "_native_metadata", lambda: {"runtime": "test", "models": {}})
    with pytest.raises(ValueError, match="no instalado"):
        lab.start(config(conditions=["no_context"], models=["unit-model"]))
    lab.engine.settings = replace(lab.engine.settings, base_url="https://ollama.com")
    with pytest.raises(ValueError, match="local"):
        lab.start(config())


def test_invalid_config_and_unknown_review_source_rejected(lab):
    for changes in [
        {"retriever": []},
        {"models": ["unknown"]},
        {"conditions": ["invalid"]},
        {"case_ids": ["unit-es"] * 2},
    ]:
        with pytest.raises(ValueError):
            lab._validate_config(config(**changes), load_dataset(lab.root))
    state = lab.start(config())
    lab._thread.join(5)
    record = lab.read_run(state["run_id"])["records"][0]
    with pytest.raises(ValueError):
        lab.review_record(
            {
                "run_id": state["run_id"],
                "record_id": record["id"],
                "claims": [{"text": "claim", "judgment": "supported", "source_ids": ["S999"]}],
            }
        )
    lab.review_record(
        {
            "run_id": state["run_id"],
            "record_id": record["id"],
            "notes": "Revisi\u00f3n",
            "claims": [{"text": "claim", "judgment": "supported", "source_ids": ["R1"]}],
        }
    )
    assert lab.read_run(state["run_id"])["records"][0]["human_review"]["claims"]
    with pytest.raises(ValueError):
        lab.read_run("../../.env")


def test_lab_http_surface_limits_origin_and_paths(lab):
    server = build_server(lab.engine, 0, lab)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            assert client.get("/lab").status_code == 200
            assert client.get("/lab.js").status_code == 200
            assert client.get("/api/lab").json()["dataset"]["revision"] == 1
            assert (
                client.post(
                    "/api/lab/start", json=config(), headers={"Origin": "https://external.invalid"}
                ).status_code
                == 403
            )
            assert client.get("/api/lab/runs/not-a-run").status_code == 400
            assert client.post("/api/lab/start", json=[]).status_code == 400
            assert client.get("/.env").status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_bm25_retrieval_does_not_load_encoder(tmp_path):
    engine = object.__new__(ChatEngine)
    engine.root, engine.document, engine.run_id = tmp_path, {"id": "synthetic"}, "unit"
    engine.specs = {"unit": {"query_prefix": "", "encoder_max_tokens": 512}}
    engine._load_encoder = lambda *_: pytest.fail("BM25 must not load GPU encoder")
    row = {
        "chunk_id": "unit",
        "parent_chunk_id": "unit",
        "text": "synthetic term",
        "pages": [1],
        "spans": [{"page_label": "1"}],
    }
    engine.collections = {("unit", "structural_512"): ([row], np.ones((1, 1)))}
    sources, metrics = engine._retrieve(
        "synthetic", [], "unit", "structural_512", method="bm25", include_trace=True
    )
    assert sources[0]["score"] is None and metrics["retrieval_method"] == "bm25"
    assert metrics["retrieval_trace"]["candidates"][0]["cosine"] is None
