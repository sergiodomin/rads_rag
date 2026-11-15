from copy import deepcopy

from radiology_rag.generation.scope import scope_comparison


def scope_fixture():
    profiles = [{"system": system, "document_id": system, "source_sha256": "unit-hash",
                 "organ": {"en": organ, "es": translated, "evidence_patterns": [organ]},
                 "modalities": [{"en": "MRI", "es": "resonancia magnética",
                                 "evidence_patterns": [r"\bMRI\b"]}]}
                for system, organ, translated in (("PI-RADS", "prostate", "próstata"),
                                                   ("BI-RADS", "breast", "mama"))]
    sources = [{"id": f"S{i + 1}", "document_id": p["document_id"],
                "source_sha256": "unit-hash", "text": p["organ"]["en"] + " MRI"}
               for i, p in enumerate(profiles)]
    return profiles, sources


def test_scope_comparison_uses_only_verified_supported_fields_and_real_citations():
    profiles, sources = scope_fixture()
    result = scope_comparison(sources, profiles, {"PI-RADS", "BI-RADS"}, "dime diferencias")
    assert "próstata" in result["answer"] and "mama" in result["answer"]
    assert result["cited_sources"] == ["S1", "S2"]
    assert all(m["sources"] for item in result["field_evidence"] for m in item["modalities"])
    english = scope_comparison(sources, profiles, {"PI-RADS", "BI-RADS"}, "difference between")
    assert "prostate" in english["answer"] and "breast" in english["answer"]


def test_scope_does_not_fill_missing_fields_or_accept_changed_provenance():
    profiles, sources = scope_fixture()
    changed = deepcopy(sources)
    changed[0]["source_sha256"] = "different"
    assert scope_comparison(changed, profiles, {"PI-RADS", "BI-RADS"}, "diferencias") is None
    changed = deepcopy(sources)
    changed[0]["text"] = "unrelated MRI"
    assert scope_comparison(changed, profiles, {"PI-RADS", "BI-RADS"}, "diferencias") is None
    assert scope_comparison(sources, profiles, {"LI-RADS"}, "diferencias") is None


def test_scope_omits_modalities_without_retrieved_evidence_and_checks_xml_identity():
    profiles, sources = scope_fixture()
    profiles[0]["modalities"].append({"en": "CT", "es": "TC", "evidence_patterns": [r"\bCT\b"]})
    result = scope_comparison(sources, profiles, {"PI-RADS", "BI-RADS"}, "diferencias")
    assert "TC" not in result["answer"]
    profiles[0]["xml_sha256"] = "expected-xml"
    assert scope_comparison(sources, profiles, {"PI-RADS", "BI-RADS"}, "diferencias") is None
