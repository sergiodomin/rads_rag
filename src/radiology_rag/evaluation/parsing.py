import re
import unicodedata


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(a, b) -> int:
    """Levenshtein, O(len(a)*len(b)), memoria lineal; para microreferencias."""
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def extraction_scores(expected: str, actual: str) -> dict:
    reference, observed = normalize(expected), normalize(actual)
    return {
        "exact_normalized": reference == observed,
        "character_error_rate": edit_distance(reference, observed) / max(1, len(reference)),
        "word_error_rate": edit_distance(reference.split(), observed.split())
        / max(1, len(reference.split())),
        "expected_characters": len(reference), "observed_characters": len(observed),
    }


def locate_anchor(text: str, anchor: str):
    """Alineación sin espacios; conserva mapeo a offsets originales, sin corregir palabras."""
    normalized, positions = [], []
    for index, char in enumerate(text):
        for expanded in unicodedata.normalize("NFKC", char).casefold():
            if not expanded.isspace():
                normalized.append(expanded)
                positions.append(index)
    target = re.sub(r"\s+", "", unicodedata.normalize("NFKC", anchor).casefold())
    if not target:
        return None
    start = "".join(normalized).find(target)
    if start < 0:
        return None
    return positions[start], positions[start + len(target) - 1] + 1


def evidence_preservation(chunks, text, cases, blocks=None):
    """Prueba de contención textual de anclas; no Recall@k ni exactitud de respuesta."""
    results = []
    for case in cases:
        spans = []
        for anchor in case["anchors"]:
            if isinstance(anchor, str):
                spans.append(locate_anchor(text, anchor))
                continue
            if blocks is None:
                raise ValueError("Las anclas con página requieren bloques con procedencia.")
            matches = []
            for block in blocks:
                if block["page_number"] != anchor["page_number"]:
                    continue
                if anchor.get("region_id") and block["region_id"] != anchor["region_id"]:
                    continue
                span = locate_anchor(block["text"], anchor["text"])
                if span:
                    matches.append((block["start"] + span[0], block["start"] + span[1]))
            spans.append(matches[0] if len(matches) == 1 else None)
        found = all(span is not None for span in spans)
        containing = [c for c in chunks if found and all(
            c["start"] <= span[0] and c["end"] >= span[1] for span in spans
        )]
        results.append({
            "case_id": case["id"], "all_anchors_in_parsed_text": found,
            "single_chunk_contains_evidence": bool(containing),
            "single_chunk_within_encoder_window": any(
                not c["exceeds_encoder_window"] for c in containing
            ),
            "note": case["note"],
        })
    return results
