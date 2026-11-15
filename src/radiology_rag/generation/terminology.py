"""Glosario anatómico mínimo; no sustituye una evaluación de fidelidad."""
import re

from radiology_rag.retrieval.hybrid import fold

# Linguistic equivalents, not classification rules. References: NCI dictionary,
# https://www.cancer.gov/espanol/publicaciones/diccionarios/diccionario-cancer
ANATOMY = {
    "prostate": {"es": "próstata", "forms": ("prostate", "prostata", "prostatico", "prostatica")},
    "breast": {"es": "mama", "forms": ("breast", "breasts", "mama", "mamario", "mamaria")},
    "penis": {"es": "pene", "forms": ("penis", "pene")},
}


def mentioned_anatomy(text):
    words = set(re.findall(r"[a-z]+", fold(text)))
    return {name for name, entry in ANATOMY.items() if words.intersection(entry["forms"])}


def source_glossary(sources):
    return [{"source_id": source["id"], "term": term, "es": ANATOMY[term]["es"]}
            for source in sources for term in sorted(mentioned_anatomy(
                source.get("document", "") + "\n" + source.get("text", "")))]


def unsupported_anatomy(answer, sources):
    available = {item["term"] for item in source_glossary(sources)}
    return mentioned_anatomy(answer) - available
