"""Un único generador recibe evidencia textual e imágenes del pool común."""

import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

import radiology_rag.fusion as module
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.fusion import FusedRAG, pack_context, parse_response
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.routing import UnifiedChat


@pytest.fixture
def fused(tmp_path, monkeypatch):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"synthetic source")
    png = tmp_path / "image.png"
    png.write_bytes(b"registered-test-image")
    folder = tmp_path / "data/processed/multimodal/v1"
    atomic_json(folder / "manifest.json", {})
    atomic_json(tmp_path / "data/catalog/vision_generator.v1.json", {})
    asset = {
        "id": "map",
        "document_id": "unit",
        "system": "PI-RADS",
        "kind": "figure",
        "source_sha256": sha256_file(pdf),
        "xml_sha256": None,
        "image_size": [100, 100],
        "title": "Synthetic map",
        "keywords": "synthetic mapa color PZ",
        "caption": "synthetic legend",
        "pdf_extracted_text": "synthetic text",
    }

    def source(i):
        return {
            "id": "C" + str(i),
            "chunk_id": "text-" + str(i),
            "parent_chunk_id": "p" + str(i),
            "document_id": "unit",
            "document": "Synthetic",
            "system": "PI-RADS",
            "pages": [i],
            "printed_pages": [str(i)],
            "text": "synthetic passage " + str(i),
            "source_sha256": asset["source_sha256"],
            "sections": [],
            "exact_category_match": False,
        }

    texts = [source(i) for i in range(1, 5)]
    calls = []
    engine = SimpleNamespace(
        root=tmp_path,
        settings=Settings(base_url="http://127.0.0.1:11434", model="old-text-model"),
        _lock=threading.Lock(),
        _phase="ready",
        documents={
            "unit": {
                "system": "PI-RADS",
                "local_path": "manual.pdf",
                "expected_sha256": asset["source_sha256"],
            }
        },
        document_collections={"unit": ["structural_512"]},
        chat_retrieval={"reranker": "unit-reranker", "candidate_count": 20},
        release_encoder=lambda: calls.append("release"),
    )

    def retrieve(*_, **options):
        assert options["reranker"] == "none"  # No text-only reranking before fusion.
        return texts, {
            "retrieval_trace": {"candidate_sources": deepcopy(texts)},
            "normalized_question": "synthetic question",
            "comparison": False,
        }

    engine._retrieve = retrieve
    engine.respond = lambda *_: pytest.fail("The legacy generator must not be called")
    visual = SimpleNamespace(
        root=tmp_path,
        folder=folder,
        index=lambda: {"assets": [asset]},
        metadata=lambda: {"assets": [asset]},
        image_path=lambda _: png,
        spec=lambda: {"model": "common-vlm", "ollama_digest": "digest", "max_image_edge": 1008},
        budget=lambda *_: {"budgeted_total_tokens": 2000},
        source=lambda _: {**source(9), "asset_id": "map", "title": "Map"},
    )

    def generate(spec, messages, *, response_format=None):
        assert engine._lock.locked() and calls.count("release") > 0
        calls.append(deepcopy(messages))
        context = json.loads(messages[-1]["content"])
        images = context["image_source_ids"]
        texts = [f["id"] for f in context["document_fragments"] if f["evidence_type"] == "text"]
        structured = {
            "abstain": False,
            "answer": "synthetic answer",
            "image_source_id": images[0] if images else "none",
            "text_source_ids": texts[:1],
        }
        assert response_format["properties"]["abstain"]["type"] == "boolean"
        return SimpleNamespace(
            content=json.dumps(structured),
            done_reason="stop",
            metrics=lambda: {"model": "common-vlm", "latency_seconds": 1},
        )

    visual.generate = generate

    class Scorer:
        def __init__(self, *_):
            pass

        def score(self, query, sources):
            assert len(sources) == 5
            assert any(s["evidence_type"] == "image" for s in sources)
            assert all(isinstance(s["text"], str) for s in sources)
            # The image proxy and three texts win the common ranking.
            return [0.8, 0.7, 0.6, 0.1, 0.9], [{}] * 5, {"pairs": 5}

    monkeypatch.setattr(module, "model_spec", lambda *_: {})
    monkeypatch.setattr(module, "LocalReranker", Scorer)
    return FusedRAG(engine, visual), calls, asset


def request(**changes):
    return {
        "message": "synthetic",
        "system": "all",
        "encoder": "e5-base",
        "collection": "structural_512",
        **changes,
    }


def test_single_pool_is_reranked_and_single_vlm_receives_both_modalities(fused):
    rag, calls, _ = fused
    result = rag.respond(request(), {"modality": "text", "reason": "normal_question"})
    assert result["metrics"]["model"] == "common-vlm"
    assert result["metrics"]["text_sources"] == 3 and result["metrics"]["images_sent"] == 1
    assert result["trace"]["image_candidate_count"] == 1
    assert result["trace"]["text_candidate_count"] == 4
    message = next(c for c in calls if isinstance(c, list))[-1]
    context = json.loads(message["content"])
    assert len(context["document_fragments"]) == 4 and len(message["images"]) == 1
    assert context["image_source_ids"] == ["S1"]
    assert [s["id"] for s in result["sources"]] == ["S1", "S2", "S3", "S4"]
    saved = next((rag.root / "output/unified/runs").glob("*.json")).read_text(encoding="utf-8")
    assert '"images":' not in saved
    assert not rag.engine._lock.locked() and rag.engine._phase == "ready"


def test_unified_entry_uses_same_generator_for_a_plain_text_question(fused, monkeypatch):
    rag, calls, _ = fused
    monkeypatch.setattr(
        rag.visual, "select", lambda *_: pytest.fail("No keyword visual branch"), raising=False
    )
    result = UnifiedChat(rag.engine, rag.visual).respond(request())
    assert result["metrics"]["model"] == "common-vlm"
    assert result["routing"]["strategy"] == "shared_candidate_pool_v1"


def test_explicit_text_control_uses_same_vlm_without_image(fused, monkeypatch):
    rag, calls, _ = fused
    monkeypatch.setattr(
        module,
        "LocalReranker",
        lambda *_: SimpleNamespace(
            score=lambda q, s: ([0.9, 0.8, 0.7, 0.6], [{}] * 4, {"pairs": 4})
        ),
    )
    result = rag.respond(request(modality="text"), {"modality": "text", "reason": "control"})
    assert result["metrics"]["model"] == "common-vlm" and result["metrics"]["images_sent"] == 0
    assert all("images" not in c[-1] for c in calls if isinstance(c, list))


def test_visual_question_reserves_its_required_image_even_if_score_is_low(fused, monkeypatch):
    rag, _, _ = fused
    monkeypatch.setattr(
        module,
        "LocalReranker",
        lambda *_: SimpleNamespace(
            score=lambda q, s: ([0.9, 0.8, 0.7, 0.6, 0.01], [{}] * 5, {"pairs": 5})
        ),
    )
    result = rag.respond(request(), {"asset_id": "map", "reason": "visual_evidence_required"})
    assert result["sources"][0]["asset_id"] == "map"
    assert result["trace"]["required_visual_asset"] == "map"


def test_packing_drops_whole_sources_not_table_notes():
    candidates = [
        {"chunk_id": "long", "text": "long table and notes", "evidence_type": "text"},
        {"chunk_id": "short", "text": "complete short source", "evidence_type": "text"},
    ]

    def budget(messages, image):
        if "long table" in str(messages):
            raise ValueError("presupuesto exceeded")
        return {"tokens": 100}

    selected, _, _, excluded = pack_context(candidates, lambda s: s, budget)
    assert selected == [{**candidates[1], "id": "S1"}]
    assert excluded == [{"chunk_id": "long", "reason": "context_budget"}]


def test_failed_generation_releases_lock_and_never_falls_back(fused):
    rag, _, _ = fused

    def fail(*_, **__):
        raise ValueError("synthetic model failure")

    rag.visual.generate = fail
    with pytest.raises(ValueError, match="model failure"):
        rag.respond(request(), {"reason": "test"})
    assert not rag.engine._lock.locked() and rag.engine._phase == "ready"


def test_invalid_citation_keeps_raw_output_but_blocks_answer(fused):
    rag, _, _ = fused
    rag.visual.generate = lambda *_, **__: SimpleNamespace(
        content=json.dumps(
            {
                "abstain": False,
                "answer": "unsupported attribution",
                "image_source_id": "S99",
                "text_source_ids": [],
            }
        ),
        done_reason="stop",
        metrics=lambda: {"model": "common-vlm", "latency_seconds": 1},
    )
    result = rag.respond(request(), {"reason": "test"})
    assert result["abstained"] and "S99" in result["raw_answer"]
    assert result["abstention_reason"] == "invalid_or_missing_citations"


def test_followup_keeps_image_but_a_semantic_topic_change_does_not(fused):
    rag, _, asset = fused
    rag.visual.select = lambda question, system, identifier: (asset, {})
    router = UnifiedChat(rag.engine, rag.visual)
    plan = router.plan(request(message="¿Y la TZ?", visual_context="map"))
    assert plan["asset_id"] == "map" and plan["reason"] == "visual_followup"
    changed = router.plan(
        request(message="¿Y qué seguimiento describe BI-RADS 3?", visual_context="map")
    )
    assert changed["modality"] == "text" and changed["context"] == ""


def test_structured_output_rejects_unknown_ids_and_malformed_json():
    sources = [{"id": "S1", "evidence_type": "image"}, {"id": "S2", "evidence_type": "text"}]
    content = json.dumps(
        {"abstain": False, "answer": "a claim", "image_source_id": "P1", "text_source_ids": []}
    )
    assert parse_response(content, sources)[2] == "invalid_or_missing_citations"
    assert parse_response('{"abstain":', sources)[2] == "invalid_structured_response"


def test_structured_abstention_does_not_publish_claims():
    content = json.dumps(
        {"abstain": True, "answer": "", "image_source_id": "none", "text_source_ids": []}
    )
    assert parse_response(content, [])[2] == "insufficient_evidence"


@pytest.mark.parametrize(
    "question",
    [
        "¿Qué es BI-RADS 3?",
        "What follow-up is described?",
        "¿Qué dice la columna de probabilidad de la tabla?",
    ],
)
def test_semantic_table_queries_do_not_require_an_image(fused, question):
    rag, _, _ = fused
    assert UnifiedChat(rag.engine, rag.visual).plan(request(message=question))["modality"] == "text"
