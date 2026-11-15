import numpy as np
import pytest

from radiology_rag.retrieval.hybrid import (
    bm25_scores,
    category_match,
    comparison_queries,
    comparison_search,
    hybrid_search,
    normalize_question,
    overview_comparison,
    query_intent,
)


@pytest.mark.parametrize("variant", ["birdas 3", "birads3", "bi rads 3", "BI-RADS 3"])
def test_known_system_names_are_normalized_without_changing_category(variant):
    normalized, _ = normalize_question("No identificar " + variant + "; 5 mm")
    assert normalized == "No identificar BI-RADS 3; 5 mm"
    assert query_intent(normalized) == ({"BI-RADS"}, {"BI-RADS": {"3"}})


@pytest.mark.parametrize("variant,canonical", [
    ("pirads", "PI-RADS"), ("PIRADS", "PI-RADS"), ("pi rads", "PI-RADS"),
    ("PI - RADS", "PI-RADS"), ("pi_rads", "PI-RADS"), ("PI-RADS", "PI-RADS"),
    ("PI‑RADS", "PI-RADS"), ("PI–RADS", "PI-RADS"),
    ("birads", "BI-RADS"), ("BIRADS", "BI-RADS"), ("bi rads", "BI-RADS"),
    ("BI - RADS", "BI-RADS"), ("bi_rads", "BI-RADS"), ("BI-RADS", "BI-RADS"),
    ("BI‑RADS", "BI-RADS"), ("BI–RADS", "BI-RADS"),
])
def test_both_families_support_spelling_case_spacing_and_unicode_hyphens(variant, canonical):
    original = f"No identificar {variant}3; lesión de 5 mm, versión 2025"
    normalized, _ = normalize_question(original)
    assert normalized == f"No identificar {canonical} 3; lesión de 5 mm, versión 2025"
    assert query_intent(normalized) == ({canonical}, {canonical: {"3"}})
    assert normalize_question(normalized) == (normalized, [])


def test_aliases_do_not_change_unrelated_words_or_merge_systems():
    text = "antibirads, biradsology, XPIRADS y lesiones de 3 mm"
    assert normalize_question(text) == (text, [])
    normalized, _ = normalize_question("pirads 4 frente a birads 3")
    assert normalized == "PI-RADS 4 frente a BI-RADS 3"
    assert query_intent(normalized)[1] == {"PI-RADS": {"4"}, "BI-RADS": {"3"}}


def test_versions_are_not_categories_and_cross_system_categories_stay_separate():
    assert query_intent("BI-RADS v2025 compared with 2013")[1] == {}
    assert query_intent("BI-RADS 2025")[1] == {}
    assert query_intent("BI-RADS 3 frente a 4")[1] == {"BI-RADS": {"3", "4"}}
    assert query_intent("BI-RADS 3 vs PI-RADS 4")[1] == {"BI-RADS": {"3"}, "PI-RADS": {"4"}}
    assert query_intent("categoría 3 en BI-RADS")[1] == {"BI-RADS": {"3"}}
    assert normalize_question("no cambiar 3 a 6 ni 2025")[0] == "no cambiar 3 a 6 ni 2025"


def test_bm25_uses_terms_and_rrf_does_not_drop_lexical_or_structured_candidates():
    assert bm25_scores("specific", ["generic", "specific detail"])[1] > 0
    assert np.all(bm25_scores("missing", ["a", "b"]) == 0)
    rows = [{"system": "BI-RADS", "text": "generic history", "assessment_categories": []},
            {"system": "BI-RADS", "text": "specific", "assessment_categories": ["3"]},
            {"system": "PI-RADS", "text": "specific", "assessment_categories": ["3"]}]
    vectors = np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32)
    ranked, cosine, lexical, _, matches = hybrid_search(
        "specific BI-RADS 3", np.array([[1, 0]]), rows, vectors,
        {"BI-RADS": {"3"}}, candidates=1)
    assert ranked[0] == 1 and matches == [False, True, False]
    assert cosine[0] > cosine[1] and lexical[1] > lexical[0]
    assert set(ranked) == {0, 1}
    assert category_match({"system": "BI-RADS", "assessment_categories": ["4A"]},
                          {"BI-RADS": {"4"}})
    assert not category_match({"system": "BI-RADS", "assessment_categories": ["4"]},
                              {"BI-RADS": {"4A"}})


def test_ablations_do_not_apply_hybrid_category_boost():
    rows = [{"text": "lexical", "assessment_categories": []},
            {"text": "other", "assessment_categories": ["3"]}]
    matrix = np.eye(2, dtype=np.float32)
    for method in ("dense", "bm25"):
        ranked, *_ = hybrid_search("lexical", matrix[:1], rows, matrix,
                                    {"PI-RADS": {"3"}}, method=method)
        assert ranked[0] == 0


def test_an_explicit_category_excludes_other_structured_categories_but_keeps_prose():
    rows = [{"system": "BI-RADS", "text": "category", "assessment_categories": ["3"]},
            {"system": "BI-RADS", "text": "category", "assessment_categories": ["4"]},
            {"system": "BI-RADS", "text": "category explanation"},
            {"system": "PI-RADS", "text": "category", "assessment_categories": ["4"]}]
    vectors = np.array([[1, 0]] * 4, dtype=np.float32)
    ranked, *_ = hybrid_search("category", vectors[:1], rows, vectors, {"BI-RADS": {"3"}})
    assert ranked[0] == 0 and set(ranked) == {0, 2, 3}


def test_comparison_allocates_two_unique_parents_per_system_even_with_biased_scores():
    rows = [{"system": system, "parent_chunk_id": f"{system}-{i}", "text": "unit"}
            for system in ("PI-RADS", "BI-RADS") for i in range(3)]
    matrix = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=np.float32)
    queries = comparison_queries("diferencia entre BI-RADS y PI-RADS", {"PI-RADS", "BI-RADS"}, {})
    vectors = {name: np.array([[1, 0]], dtype=np.float32) for name in queries}
    ranked, *_ = comparison_search(queries, vectors, rows, matrix, {})
    assert len(ranked) == 4
    assert [rows[i]["system"] for i in ranked] == ["BI-RADS", "PI-RADS"] * 2
    with pytest.raises(ValueError, match="Falta un documento"):
        comparison_search(queries, vectors, rows[:3], matrix[:3], {})


def test_general_comparisons_use_scope_sections_but_specific_questions_keep_all_candidates():
    assert overview_comparison("diferencia entre BI-RADS y PI-RADS", {})
    assert not overview_comparison("BI-RADS vs PI-RADS MRI contrast", {})
    assert not overview_comparison("BI-RADS vs PI-RADS", {"BI-RADS": {"3"}})
    rows = [{"system": system, "parent_chunk_id": f"{system}-{section}",
             "section_paths": [[section]], "text": "scope unit"}
            for system in ("PI-RADS", "BI-RADS") for section in ("Lexicon", "Introduction")]
    matrix = np.array([[1, 0]] * len(rows), dtype=np.float32)
    queries = comparison_queries("difference BI-RADS PI-RADS", {"PI-RADS", "BI-RADS"}, {})
    vectors = {name: matrix[:1] for name in queries}
    ranked, *_ = comparison_search(queries, vectors, rows, matrix, {}, overview=True)
    assert set(ranked) == {1, 3}


@pytest.mark.parametrize("question", [
    "dime las diferencias entre birads y pirads",
    "explícame las diferencias entre BI-RADS y PI-RADS, por favor",
    "¿Puedes decir cuáles son las diferencias entre BI-RADS y PI-RADS?",
    "cuéntame las principales diferencias entre BI-RADS y PI-RADS",
])
def test_overview_paraphrases_do_not_drop_scope_evidence(question):
    normalized, _ = normalize_question(question)
    # The screenshot's wording and other common framing retain overview routing.
    assert overview_comparison(normalized, {})
    rows = [{"system": system, "parent_chunk_id": f"{system}-{section}",
             "section_paths": [[section]], "text": "unit"}
            for system in ("PI-RADS", "BI-RADS") for section in ("Lexicon", "Introduction")]
    matrix = np.array([[1, 0]] * len(rows), dtype=np.float32)
    queries = comparison_queries(normalized, {"PI-RADS", "BI-RADS"}, {})
    vectors = {name: matrix[:1] for name in queries}
    # Even deliberately disabling overview classification must preserve both anchors.
    ranked, *_ = comparison_search(queries, vectors, rows, matrix, {}, overview=False)
    assert ranked[:2] == [3, 1]
    assert set(ranked) == {0, 1, 2, 3}
