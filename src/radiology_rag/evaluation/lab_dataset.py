"""Dataset local revisable; las referencias propuestas no son un gold clínico."""

import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime

from radiology_rag.chat import validate_request
from radiology_rag.evaluation.parsing import locate_anchor
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import within
from radiology_rag.retrieval.dense import read_jsonl

DATASET = "data/evaluations/lab.v1.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reference_source(engine, document_id, row):
    document = engine.documents[document_id]
    pages = row.get("pages", [row["page_number"]] if "page_number" in row else [])
    spans = row.get("spans", [{"page_label": row.get("page_label", str(p))} for p in pages])
    return {
        "id": "S1",
        "document_id": document_id,
        "system": document.get("system", "PI-RADS"),
        "document": document["title"],
        "document_role": document["role"],
        "document_version": document["version"],
        "source_sha256": row["source_sha256"],
        "xml_sha256": row.get("xml_sha256"),
        "pages": pages,
        "printed_pages": list(dict.fromkeys(s["page_label"] for s in spans)),
        "sections": row.get("section_paths", [row.get("section_path", [])]),
        "chunk_id": row.get("chunk_id", row.get("block_id", fingerprint(row))),
        "text": row.get("parent_text") or row["text"],
        "xml_element_id": row.get("xml_element_id"),
        "table_row": row.get("table_row"),
        "assessment_categories": row.get("assessment_categories", []),
        "table_headers": row.get("table_headers", []),
        "table_cells": row.get("table_cells", []),
        "pdf_url": (
            f"/documents/{document_id}.pdf#page={pages[0]}" if pages else document["source_url"]
        ),
    }


def evidence_matches(source, evidence):
    expected = evidence["source"]
    if (
        source.get("document_id") != expected["document_id"]
        or source.get("source_sha256") != expected["source_sha256"]
    ):
        return False
    if expected.get("xml_sha256") and source.get("xml_sha256") != expected["xml_sha256"]:
        return False
    selector = evidence["selector"]
    if selector.get("page_number") and selector["page_number"] not in source.get("pages", []):
        return False
    for key in ("xml_element_id", "table_row"):
        if key in selector and source.get(key) != selector[key]:
            return False
    return (
        not selector.get("anchor") or locate_anchor(source["text"], selector["anchor"]) is not None
    )


def oracle_sources(case):
    result, seen = [], set()
    for item in case["evidence"]:
        source = deepcopy(item["source"])
        key = source["document_id"], source["chunk_id"]
        if key not in seen:
            source["id"] = f"S{len(result) + 1}"
            result.append(source)
            seen.add(key)
    return result


def validate_dataset(data):
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("Dataset de laboratorio no válido.")
    ids, groups = set(), {}
    for case in data["cases"]:
        validate_request({"message": case["question"], "history": case.get("history", [])})
        if case["id"] in ids or case["split"] not in {"dev", "test"}:
            raise ValueError("IDs duplicados o partición no válida.")
        ids.add(case["id"])
        previous = groups.setdefault(case["group_id"], case["split"])
        if previous != case["split"]:
            raise ValueError("Un grupo de paráfrasis/traducciones cruza dev y test.")
        if case["answerable"] and not case["evidence"]:
            raise ValueError("Un caso respondible necesita evidencia de referencia.")
        if len({e["id"] for e in case["evidence"]}) != len(case["evidence"]):
            raise ValueError("IDs de evidencia duplicados.")
        if case["review_status"] not in {"draft", "reviewed"}:
            raise ValueError("Estado de revisión desconocido.")
        if (
            case["answerable"]
            and case["review_status"] == "reviewed"
            and not case["reference_answer"].strip()
        ):
            raise ValueError("Una referencia revisada respondible necesita respuesta.")
        for item in case["evidence"]:
            if not evidence_matches(item["source"], item):
                raise ValueError("Referencia no resoluble en " + case["id"])
    return data


def load_dataset(root):
    path = root / DATASET
    if not path.exists():
        raise ValueError("Prepara primero el dataset con scripts/prepare_lab_dataset.py.")
    return validate_dataset(json.loads(path.read_text(encoding="utf-8")))


def resolve_seed(blocks, item):
    candidates = [
        b
        for b in blocks
        if b["block_id"] == item["block_id"] and b["page_number"] == item["page_number"]
    ]
    if len(candidates) != 1:
        raise ValueError("Bloque del seed ausente o ambiguo; revisar el parsing.")
    block = candidates[0]
    if fingerprint(block["text"]) != item["block_text_sha256"]:
        raise ValueError("El texto local difiere del seed; revisar los offsets.")
    if not 0 <= item["start"] < item["end"] <= len(block["text"]):
        raise ValueError("Offsets del seed fuera del bloque.")
    return block, {
        "page_number": item["page_number"],
        "anchor": block["text"][item["start"] : item["end"]],
    }


def prepare_dataset(engine):
    root = engine.root
    if (root / DATASET).exists():
        return load_dataset(root)
    pi, bi = "pirads-2.1-2019", "birads-v2025-review-2026"
    if not {pi, bi} <= engine.documents.keys():
        raise ValueError("Prepara los índices PI-RADS y BI-RADS antes del laboratorio.")
    pointer = json.loads((root / f"data/processed/{pi}/02_chunking/latest.json").read_text())
    blocks = read_jsonl(within(root, pointer["path"]) / "located_blocks.jsonl")
    bi_rows = [
        r for r in engine.collections["qwen3-0.6b", "structural_512"][0] if r["document_id"] == bi
    ]
    cases = []

    def unit(document_id, row, selector):
        return {
            "id": fingerprint([document_id, selector])[:16],
            "selector": selector,
            "source": reference_source(engine, document_id, row),
        }

    def group(identifier, questions, evidence, kind, *, scope="all", split="dev", history=None):
        for language, question in questions.items():
            case = {
                "id": f"{identifier}-{language}",
                "group_id": identifier,
                "split": split,
                "language": language,
                "question": question,
                "type": kind,
                "system": scope,
                "answerable": bool(evidence),
                "evidence": deepcopy(evidence),
                "reference_answer": "\n\n".join(
                    s["text"] for s in oracle_sources({"evidence": evidence})
                ),
                "reference_language": "en",
                "review_status": "draft",
                "review_notes": "",
                "history": (history or {}).get(language, []),
            }
            cases.append(case)

    pilot = json.loads((root / "data/templates/lab_seed.v1.json").read_text(encoding="utf-8"))
    if pilot["source_sha256"] != engine.documents[pi]["expected_sha256"]:
        raise ValueError("El seed corresponde a otro PDF.")
    for old in pilot["cases"]:
        if old["id"] == "breast-out-of-domain":
            continue  # Breast is no longer outside this multi-corpus system.
        evidence = []
        for item in old["evidence"]:
            block, selector = resolve_seed(blocks, item)
            evidence.append(unit(pi, block, selector))
        group("pi-" + old["id"], old["questions"], evidence, old["type"], scope="PI-RADS")
    group(
        "unavailable-astronomy",
        {
            "es": "¿Qué periodo orbital de Marte recoge esta documentación?",
            "en": "What orbital period of Mars is recorded in these documents?",
        },
        [],
        "out_of_domain",
    )

    def bi_evidence(predicate):
        row = next(r for r in bi_rows if predicate(r))
        selector = {"xml_element_id": row["xml_element_id"]}
        if row.get("table_row") is not None:
            selector["table_row"] = row["table_row"]
        else:
            selector["anchor"] = row["text"].split("\n", 1)[-1][:100]
        return [unit(bi, row, selector)]

    for category in ("3", "6"):
        evidence = bi_evidence(lambda r, c=category: r.get("assessment_categories") == [c])
        group(
            "bi-category-" + category,
            {
                "es": (
                    f"¿Qué describe la revisión para BI-RADS categoría {category} "
                    "y qué límites indica?"
                ),
                "en": (
                    f"What does the review describe for BI-RADS category {category}, "
                    "including limits?"
                ),
            },
            evidence,
            "table",
            scope="BI-RADS",
        )
    sections = [
        (
            "structure",
            "sec3-",
            "¿Qué cambios de estructura describe la revisión para BI-RADS v2025?",
            "What structural changes does the review describe for BI-RADS v2025?",
            "dev",
        ),
        (
            "cem",
            "sec7-",
            "¿Cómo describe la revisión el lugar de CEM en BI-RADS v2025?",
            "How does the review describe the place of CEM in BI-RADS v2025?",
            "dev",
        ),
        (
            "nonmass",
            "sec5dot2-",
            "Localiza los cambios sobre lesiones no masa en ecografía.",
            "Find the changes concerning non-mass lesions in ultrasound.",
            "test",
        ),
        (
            "rind",
            "sec5dot4-",
            "¿Qué cambio describe la revisión sobre el descriptor echogenic rind?",
            "What change does the review describe for the echogenic rind descriptor?",
            "test",
        ),
        (
            "implants",
            None,
            "Localiza la discusión de los descriptores de implantes en RM.",
            "Find the discussion of implant descriptors in MRI.",
            "test",
        ),
        (
            "calcifications",
            "sec4dot3-",
            "¿Qué cambios de terminología de calcificaciones se describen?",
            "What calcification terminology changes are described?",
            "test",
        ),
    ]
    for identifier, prefix, es, en, split in sections:
        evidence = bi_evidence(
            lambda r, p=prefix: (
                r.get("kind") == "prose"
                and (
                    r["xml_element_id"].startswith(p)
                    if p
                    else any("implant" in s.lower() for path in r["section_paths"] for s in path)
                )
            )
        )
        group(
            "bi-" + identifier,
            {"es": es, "en": en},
            evidence,
            "section",
            scope="BI-RADS",
            split=split,
        )

    pi_intro = next(
        b
        for b in blocks
        if b["section_path"] == ["Introduction"]
        and locate_anchor(b["text"], "prostate MRI") is not None
    )
    bi_intro = next(
        r
        for r in bi_rows
        if r.get("kind") == "prose"
        and "breasts" in r["text"]
        and "Introduction" in str(r["section_paths"])
    )
    cross = [
        unit(pi, pi_intro, {"page_number": pi_intro["page_number"], "anchor": "prostate MRI"}),
        unit(bi, bi_intro, {"xml_element_id": bi_intro["xml_element_id"], "anchor": "breasts"}),
    ]
    group(
        "cross-scope",
        {
            "es": "Compara órgano y modalidades de BI-RADS y PI-RADS.",
            "en": "Compare the organ and modalities of BI-RADS and PI-RADS.",
        },
        cross,
        "comparison",
    )
    group(
        "cross-mri",
        {
            "es": "Compara el papel de la RM en PI-RADS y en la revisión BI-RADS.",
            "en": "Compare the role of MRI in PI-RADS and in the BI-RADS review.",
        },
        cross,
        "comparison",
    )
    history = {
        lang: [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "BI-RADS 3: previous documentary topic."},
        ]
        for lang, text in {
            "es": "dime qué caracteriza un birads 3",
            "en": "describe BI-RADS category 3",
        }.items()
    }
    group(
        "conversation-topic-switch",
        {
            "es": "dime las diferencias entre birads y pirads",
            "en": "tell me the differences between birads and pirads",
        },
        cross,
        "conversation",
        history=history,
    )
    group(
        "conversation-category-followup",
        {"es": "¿Y qué cambió para la categoría 6?", "en": "And what changed for category 6?"},
        bi_evidence(lambda r: r.get("assessment_categories") == ["6"]),
        "conversation",
        history=history,
    )
    data = {
        "schema_version": 1,
        "dataset_id": "rads-lab-v1",
        "revision": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "draft_not_clinically_validated",
        "seed_index_runs": engine.run_id,
        "cases": cases,
        "notes": [
            "Legacy PI-RADS cases are development, including former pilot test cases.",
            "Breast out-of-domain control replaced after adding BI-RADS.",
            "Conversation histories are fixed fixtures, not model-generated trajectories.",
            "References are extractive drafts, awaiting human review.",
        ],
        "source_files": {
            p: sha256_file(root / p)
            for p in (
                "data/catalog/documents.v1.json",
                "data/templates/lab_seed.v1.json",
            )
        },
    }
    validate_dataset(data)
    if len(cases) != 48:
        raise ValueError("El dataset inicial debe contener 48 consultas agrupadas.")
    atomic_json(root / DATASET, data)
    return data
