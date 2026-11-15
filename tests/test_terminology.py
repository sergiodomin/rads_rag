from radiology_rag.generation.terminology import source_glossary, unsupported_anatomy


def test_glossary_contains_only_terms_located_in_retrieved_sources():
    sources = [{"id": "S1", "document": "MRI", "text": "prostate imaging"},
               {"id": "S2", "document": "Review", "text": "dense breasts"}]
    assert source_glossary(sources) == [
        {"source_id": "S1", "term": "prostate", "es": "próstata"},
        {"source_id": "S2", "term": "breast", "es": "mama"}]
    assert unsupported_anatomy("Mama y próstata", sources) == set()
    assert unsupported_anatomy("Pene", sources) == {"penis"}
    assert source_glossary([{"id": "S1", "text": "Unit unrelated text"}]) == []
