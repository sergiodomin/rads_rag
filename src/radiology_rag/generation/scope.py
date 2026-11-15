"""Comparación extractiva de campos verificados, con evidencia por campo."""
import re


def scope_comparison(sources, profiles, systems, question):
    english = bool(re.search(r"\b(?:difference|differences|compare|between)\b", question, re.I))
    language = "en" if english else "es"
    lines, used, evidence = [], set(), []
    for system in sorted(systems):
        profile = next((p for p in profiles if p["system"] == system), None)
        if profile is None:
            return None
        candidates = [s for s in sources if s.get("document_id") == profile["document_id"]
                      and s.get("source_sha256") == profile["source_sha256"]
                      and (not profile.get("xml_sha256")
                           or s.get("xml_sha256") == profile["xml_sha256"])]

        def support(field, source_rows=candidates):
            return [s["id"] for s in source_rows if any(
                re.search(pattern, s.get("document", "") + "\n" + s["text"], re.I)
                for pattern in field["evidence_patterns"])]

        organ_refs = support(profile["organ"])
        modalities = [(m, support(m)) for m in profile["modalities"]]
        modalities = [(m, refs) for m, refs in modalities if refs]
        if not organ_refs or not modalities:
            return None
        refs = {organ_refs[0], *(ids[0] for _, ids in modalities)}
        used.update(refs)
        organ_label = "organ" if english else "órgano"
        modality_label = "modalities in these excerpts" if english else "modalidades mencionadas"
        citation = " ".join(f"[{ref}]" for ref in sorted(refs))
        values = ", ".join(m[language] for m, _ in modalities)
        lines.append(f"- **{system}**: {organ_label}: **{profile['organ'][language]}**; "
                     f"{modality_label}: {values}. {citation}")
        evidence.append({"system": system, "organ_sources": organ_refs,
                         "modalities": [{"value": m[language], "sources": refs}
                                        for m, refs in modalities]})
    return {"answer": "\n\n".join(lines), "cited_sources": sorted(used),
            "field_evidence": evidence}
