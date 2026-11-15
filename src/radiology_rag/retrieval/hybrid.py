"""Consulta normalizada, BM25 y RRF con coincidencias de categoría explícitas."""
import math
import re
import unicodedata
from collections import Counter

import numpy as np

from radiology_rag.retrieval.dense import exact_search

NAME_SEPARATOR = r"[\s_\-\u2010-\u2015\u2212]*"
# One registry for aliases, canonical names and query-intent recognition.
# Adding a name does not incorporate its documents or enable a new chat filter.
SYSTEM_ALIASES = {
    "PI-RADS": (rf"pi{NAME_SEPARATOR}rads",),
    "BI-RADS": (rf"bi{NAME_SEPARATOR}rads", r"birdas"),
}


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


def normalize_question(question):
    """Sólo variantes conocidas de nombres; conserva números, negaciones y unidades."""
    text, changes = question, []
    for name, aliases in SYSTEM_ALIASES.items():
        expression = r"\b(?:" + "|".join(aliases) + r")(?=\b|[0-6])"
        def substitute(match, canonical=name):
            if match[0] != canonical:
                changes.append({"from": match[0], "to": canonical})
            return canonical
        text = re.sub(expression, substitute, text, flags=re.IGNORECASE)
    names = "|".join(re.escape(name) for name in SYSTEM_ALIASES)
    text = re.sub(r"\b(" + names + r")(?=[0-6](?:[abcABC])?\b)", r"\1 ", text)
    return text, changes


def query_intent(question):
    text = fold(question)
    systems = {name for name in SYSTEM_ALIASES
               if re.search(r"\b" + re.escape(name.lower()) + r"(?=\b|[0-6])", text)}
    categories = {}
    names = "|".join(re.escape(name.lower()) for name in SYSTEM_ALIASES)
    expression = (r"(" + names + r")\s*(?:(?:categoria|category|categories)\s*)?"
                  r"([0-6][abc]?)\b")
    for match in re.finditer(expression, text):
        system = match[1].upper()
        categories.setdefault(system, set()).add(match[2].upper())
        following = text[match.end():]
        extra = re.match(r"\s*(?:,|y|and|vs\.?|versus|frente a)\s*([0-6][abc]?)\b", following)
        if extra:
            categories[system].add(extra[1].upper())
    if len(systems) == 1 and not categories:
        values = re.findall(r"\b(?:categoria|category)\s*([0-6][abc]?)\b", text)
        if values:
            categories[next(iter(systems))] = {v.upper() for v in values}
    return systems, categories


def tokens(text):
    return re.findall(r"[a-z0-9]+", fold(text))


def bm25_scores(query, texts, *, k1=1.2, b=0.75):
    documents = [Counter(tokens(text)) for text in texts]
    lengths = [sum(d.values()) for d in documents]
    average = sum(lengths) / max(len(lengths), 1) or 1
    scores = np.zeros(len(documents), dtype=np.float64)
    for token in set(tokens(query)):
        frequency = sum(token in d for d in documents)
        if not frequency:
            continue
        idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for i, document in enumerate(documents):
            count = document[token]
            scores[i] += idf * count * (k1 + 1) / (
                count + k1 * (1 - b + b * lengths[i] / average))
    return scores


def category_match(row, requested):
    wanted = requested.get(row.get("system", "PI-RADS"), set())
    return any(found == value or (len(value) == 1 and found.startswith(value))
               for value in wanted for found in row.get("assessment_categories", []))


def hybrid_search(query, vector, rows, matrix, categories, *, candidates=50, method="hybrid"):
    """RRF combina rangos; las filas de categorías solicitadas forman el primer grupo."""
    if method not in {"hybrid", "dense", "bm25"}:
        raise ValueError("Recuperador no disponible.")
    dense_scores = np.zeros(len(rows), dtype=np.float64)
    dense_order = []
    if method != "bm25":
        positions, cosine = exact_search(vector, matrix, k=len(rows))
        dense_scores[positions[0]] = cosine[0]
        dense_order = positions[0]
    lexical = bm25_scores(query, [r.get("embedding_text", r["text"]) for r in rows])
    lexical_order = np.argsort(-lexical, kind="stable")
    fused = np.zeros(len(rows), dtype=np.float64)
    matches = [category_match(row, categories) for row in rows]
    if method == "dense":
        return list(dense_order[:candidates]), dense_scores, lexical, fused, matches
    if method == "bm25":
        return ([i for i in lexical_order if lexical[i] > 0][:candidates],
                dense_scores, lexical, fused, matches)
    for order in (dense_order[:candidates],
                  [i for i in lexical_order if lexical[i] > 0][:candidates]):
        for rank, index in enumerate(order, 1):
            fused[index] += 1 / (60 + rank)
    constrained = {row.get("system", "PI-RADS") for row, match in zip(rows, matches, strict=True)
                   if match}
    pool = {i for i, row in enumerate(rows) if (fused[i] or matches[i]) and not (
        row.get("system", "PI-RADS") in constrained and row.get("assessment_categories")
        and not matches[i])}
    ranked = sorted(pool, key=lambda i: (not matches[i], -fused[i], -dense_scores[i], i))
    return ranked, dense_scores, lexical, fused, matches


def comparison_queries(question, systems, categories):
    return {system: (f"{system} categories {', '.join(sorted(categories[system]))}. "
                     f"Definition, criteria and scope. {question}" if categories.get(system)
                    else f"{system}: definition, purpose, organ, imaging modalities, "
                    f"reporting scope. {question}") for system in sorted(systems)}


def overview_comparison(question, categories):
    if categories:
        return False
    # Request framing is not a technical topic. Retrieval still has scope anchors
    # if an unfamiliar paraphrase is not classified as an overview.
    question = re.sub(
        r"\b(?:dime|dinos|explicame|explicanos|explica|cuentame|muestrame|describe|resume|"
        r"indicame|puedes|podrias|decir|por favor|me gustaria saber|quiero saber|necesito saber)\b",
        " ", fold(question))
    general = {"bi", "pi", "rads", "diferencia", "diferencias", "entre", "y", "compara",
               "comparar", "comparacion", "los", "las", "que", "hay", "cual", "cuales", "es",
               "son", "sistema", "sistemas", "difference", "differences", "between", "and",
               "compare", "comparison", "vs", "versus", "the", "what", "is", "are",
               "principales", "basicas", "main", "basic"}
    return bool(tokens(question)) and not set(tokens(question)) - general


def comparison_search(queries, vectors, rows, matrix, categories, *, per_system=2,
                      overview=False, method="hybrid"):
    """Recuperación independiente y equilibrada; no fabrica un documento comparativo."""
    scores = np.zeros(len(rows))
    lexical, fused = scores.copy(), scores.copy()
    matches = [False] * len(rows)
    groups = []
    for system, query in queries.items():
        indices = [i for i, row in enumerate(rows) if row.get("system", "PI-RADS") == system]
        if not indices:
            raise ValueError("Falta un documento indexado para comparar " + system)
        introductions = [i for i in indices if any(
            {"introduction", "introduccion", "scope", "alcance", "purpose"}
            & set(tokens(section)) for path in rows[i].get("section_paths", [])
            for section in path)]
        if overview and introductions:
            indices = introductions
        subset = [rows[i] for i in indices]
        ranked, cos, lex, rrf, exact = hybrid_search(
            query, vectors.get(system), subset, matrix[indices], categories, method=method)
        scores[indices], lexical[indices], fused[indices] = cos, lex, rrf
        for local, index in enumerate(indices):
            matches[index] = exact[local]
        selected, seen = [], set()
        if not overview and introductions and not categories.get(system):
            # Always keep one scope anchor, regardless of question phrasing.
            anchors, *_ = hybrid_search(query, vectors.get(system),
                                        [rows[i] for i in introductions],
                                        matrix[introductions], categories, method=method)
            if anchors:
                anchor = introductions[anchors[0]]
                selected.append(anchor)
                seen.add(rows[anchor]["parent_chunk_id"])
        for local in ranked:
            parent = subset[local]["parent_chunk_id"]
            if parent not in seen:
                selected.append(indices[local])
                seen.add(parent)
            if len(selected) == per_system:
                break
        groups.append(selected)
    order = [group[rank] for rank in range(per_system) for group in groups if len(group) > rank]
    return order, scores, lexical, fused, matches
