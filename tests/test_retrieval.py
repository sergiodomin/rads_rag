import numpy as np
import pytest

from radiology_rag.evaluation.retrieval import (
    calibrate_abstention,
    evidence_in_passage,
    resolve_evidence,
    retrieval_metrics,
)
from radiology_rag.retrieval.dense import exact_search, normalize_rows


def test_cosine_ranking_ignores_magnitude_and_keeps_ties_stable():
    passages = [[4, 0], [1, 0], [0, 2], [-3, 0]]
    indices, scores = exact_search([[5, 0], [0, 9]], passages, k=10)
    assert indices.tolist() == [[0, 1, 2, 3], [2, 0, 1, 3]]
    np.testing.assert_allclose(scores, [[1, 1, 0, -1], [1, 0, 0, 0]])


@pytest.mark.parametrize("matrix", [[[0, 0]], [[float("nan"), 1]], [1, 2]])
def test_invalid_embedding_cannot_silently_enter_cosine_search(matrix):
    with pytest.raises(ValueError):
        normalize_rows(matrix)


def test_overlap_duplicates_do_not_inflate_unique_evidence_recall():
    evidence = {"a": {"unit1"}, "b": {"unit1"}, "c": {"unit2"}, "miss": set()}
    result = retrieval_metrics(["miss", "a", "b", "c"], {"a", "b", "c"}, evidence, k=3)
    assert result["recall@3"] == pytest.approx(2 / 3)
    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["mrr@3"] == 0.5
    assert result["evidence_recall@3"] == 0.5
    assert result["all_evidence@3"] == 0
    assert retrieval_metrics(["c", "a"], {"a", "b", "c"}, evidence, k=2)["all_evidence@2"] == 1


def test_page_scoped_evidence_requires_the_complete_original_span():
    blocks = [
        {"page_number": 1, "block_id": "other", "start": 0, "text": "findings next"},
        {"page_number": 2, "block_id": "target", "start": 100, "text": "ﬁndings  next"},
    ]
    case = {"id": "case", "evidence": [
        {"id": "unit", "page_number": 2, "anchor": "findings next"},
    ]}
    units = resolve_evidence(case, blocks)
    assert units[0]["start"] == 100 and units[0]["end"] == 113
    assert evidence_in_passage({"start": 100, "end": 113}, units) == {"unit"}
    assert evidence_in_passage({"start": 100, "end": 112}, units) == set()
    with pytest.raises(ValueError, match="ambigua"):
        resolve_evidence(case, [*blocks, {**blocks[1], "block_id": "duplicate"}])


def test_abstention_threshold_uses_development_and_rejects_test_leakage():
    rows = [{"split": "dev", "top_score": score, "answerable": label}
            for score, label in [(0.2, False), (0.4, False), (0.7, True), (0.8, True)]]
    calibration = calibrate_abstention(rows)
    assert calibration["threshold"] == pytest.approx(0.55)
    assert calibration["dev_balanced_accuracy"] == 1
    with pytest.raises(ValueError, match="desarrollo"):
        calibrate_abstention([*rows, {**rows[0], "split": "test"}])
