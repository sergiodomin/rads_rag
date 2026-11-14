"""Métricas deterministas sobre anclas de evidencia anotadas antes de recuperar."""
from radiology_rag.evaluation.parsing import locate_anchor


def resolve_evidence(case, blocks):
    resolved = []
    for item in case["evidence"]:
        candidates = []
        for block in blocks:
            if block["page_number"] != item["page_number"]:
                continue
            span = locate_anchor(block["text"], item["anchor"])
            if span:
                candidates.append({**item, "start": block["start"] + span[0],
                                   "end": block["start"] + span[1],
                                   "block_id": block["block_id"]})
        if len(candidates) != 1:
            raise ValueError(f"Evidencia ausente o ambigua: {case['id']} / {item['id']}")
        resolved.append(candidates[0])
    return resolved


def evidence_in_passage(passage, evidence):
    return {item["id"] for item in evidence
            if passage["start"] <= item["start"] and passage["end"] >= item["end"]}


def retrieval_metrics(ranked_ids, relevant_ids, evidence_by_id, *, k=5):
    if k < 1 or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Se requiere k positivo y un ranking sin IDs repetidos.")
    if not relevant_ids:
        raise ValueError("Recall/MRR requieren al menos un pasaje relevante anotado.")
    retrieved = ranked_ids[:k]
    relevant_ids = set(relevant_ids)
    hits = [identifier in relevant_ids for identifier in retrieved]
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    expected = set().union(*(evidence_by_id[i] for i in relevant_ids))
    found = set().union(*(evidence_by_id.get(i, set()) for i in retrieved))
    return {
        f"recall@{k}": sum(hits) / len(relevant_ids),
        f"precision@{k}": sum(hits) / len(retrieved) if retrieved else 0,
        f"mrr@{k}": 1 / first if first else 0,
        f"hit@{k}": float(any(hits)),
        f"evidence_recall@{k}": len(found & expected) / len(expected),
        f"all_evidence@{k}": float(expected <= found),
    }


def calibrate_abstention(rows):
    """Elige umbral por balanced accuracy usando exclusivamente desarrollo."""
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("La calibración sólo admite desarrollo.")
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    if not positive or not negative:
        raise ValueError("Se necesitan controles y preguntas respondibles en desarrollo.")
    scores = sorted({r["top_score"] for r in rows})
    thresholds = [scores[0] - 1e-6,
                  *[(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)],
                  scores[-1] + 1e-6]

    def balanced_accuracy(threshold):
        sensitivity = sum(r["top_score"] >= threshold for r in positive) / len(positive)
        specificity = sum(r["top_score"] < threshold for r in negative) / len(negative)
        return (sensitivity + specificity) / 2

    threshold = max(thresholds, key=lambda value: (balanced_accuracy(value), value))
    return {"threshold": threshold, "dev_balanced_accuracy": balanced_accuracy(threshold),
            "dev_positive": len(positive), "dev_negative": len(negative)}
