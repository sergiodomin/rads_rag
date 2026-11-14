from radiology_rag.evaluation.parsing import evidence_preservation, extraction_scores, locate_anchor


def test_missing_figure_text_and_merged_words_are_not_scored_as_exact():
    assert extraction_scores("Choline Citrate", "")["word_error_rate"] == 1
    assert extraction_scores("Table caption", "Tablecaption")["exact_normalized"] is False
    assert extraction_scores("Table  caption", "Table\ncaption")["exact_normalized"] is True


def test_alignment_maps_unicode_expansion_to_original_offsets():
    text = "prefix\nﬁndings  next"
    span = locate_anchor(text, "findings next")
    assert span is not None and text[span[0]:span[1]] == "ﬁndings  next"


def test_evidence_in_oversize_parent_is_not_reported_as_encoder_ready():
    text = "Table title then continuation note"
    chunks = [{"start": 0, "end": len(text), "exceeds_encoder_window": True}]
    result = evidence_preservation(chunks, text, [{"id": "test", "anchors": ["Table title", "note"],
                                                 "note": "test-only"}])[0]
    assert result["single_chunk_contains_evidence"]
    assert not result["single_chunk_within_encoder_window"]


def test_scoped_anchor_does_not_match_same_heading_on_another_page():
    text = "Repeated title\n\nRepeated title"
    blocks = [
        {"page_number": 1, "region_id": "p1", "start": 0, "text": "Repeated title"},
        {"page_number": 2, "region_id": "p2", "start": 16, "text": "Repeated title"},
    ]
    chunks = [{"start": 0, "end": 14, "exceeds_encoder_window": False}]
    case = {"id": "page2", "anchors": [{"text": "Repeated title", "page_number": 2}],
            "note": "test"}
    result = evidence_preservation(chunks, text, [case], blocks)[0]
    assert result["all_anchors_in_parsed_text"]
    assert not result["single_chunk_contains_evidence"]
    chunks.append({"start": 16, "end": len(text), "exceeds_encoder_window": False})
    assert evidence_preservation(chunks, text, [case], blocks)[0][
        "single_chunk_within_encoder_window"]


def test_missing_scoped_page_is_not_satisfied_by_global_text():
    case = {"id": "missing", "anchors": [{"text": "title", "page_number": 2}], "note": "test"}
    blocks = [{"page_number": 1, "region_id": "p1", "start": 0, "text": "title"}]
    result = evidence_preservation([], "title", [case], blocks)[0]
    assert not result["all_anchors_in_parsed_text"]
