"""Pasajes JATS con tablas trazables; las figuras permanecen sin interpretar."""
import json
import re
from xml.etree import ElementTree as ET

from radiology_rag.ingestion.parsing import fingerprint


def text_of(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def expand_rows(rows):
    """Expande rowspan/colspan conservando el valor en cada columna ocupada."""
    pending = {}
    result = []
    for row in rows:
        values = {col: value for col, (value, _) in pending.items()}
        following = {col: (value, count - 1) for col, (value, count) in pending.items()
                     if count > 1}
        col = 0
        for cell in row:
            if cell.tag not in {"td", "th"}:
                continue
            while col in values:
                col += 1
            width, height = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
            if not 1 <= width <= 100 or not 1 <= height <= 100:
                raise ValueError("Tabla JATS con spans no compatibles.")
            for offset in range(width):
                position = col + offset
                if position in values:
                    raise ValueError("Celdas solapadas en la tabla JATS.")
                values[position] = text_of(cell)
                if height > 1:
                    following[position] = (values[position], height - 1)
            col += width
        result.append([values.get(i, "") for i in range(max(values, default=-1) + 1)])
        pending = following
    return result


def table_records(wrapper):
    table = wrapper.find("table")
    if table is None:
        raise ValueError("Tabla sin celdas estructuradas.")
    headers, section, records, after_section, row_number = [], "", [], False, 0
    for group in table:
        if group.tag == "thead":
            headers = expand_rows(group.findall("tr"))
        elif group.tag in {"tbody", "tfoot"}:
            original = group.findall("tr")
            for source_row, cells in zip(original, expand_rows(original), strict=True):
                row_number += 1
                if cells and len(set(cells)) == 1:
                    section = cells[0]
                    after_section = True
                    continue
                # This publisher encodes internal header rows as bold td cells.
                internal_header = after_section and all(
                    cell.tag == "th" or (cell.find("bold") is not None
                                         and text_of(cell) == text_of(cell.find("bold")))
                    for cell in source_row if cell.tag in {"th", "td"})
                if internal_header:
                    headers = [cells]
                    after_section = False
                    continue
                after_section = False
                labels = [" / ".join(dict.fromkeys(
                    row[i] for row in headers if i < len(row) and row[i]
                )) or f"column {i + 1}" for i in range(len(cells))]
                records.append({"section": section, "cells": cells, "headers": labels,
                                "row_number": row_number})
    if not records:
        raise ValueError("Tabla JATS sin filas de datos.")
    return records


def row_attributes(record):
    categories = []
    for header, value in zip(record["headers"], record["cells"], strict=True):
        if header.split(" / ")[-1].strip().lower() in {
            "category", "assessment category", "bi-rads category",
        }:
            match = re.fullmatch(r"\s*([0-6][abcABC]?)\s*", value)
            if match:
                categories.append(match[1].upper())
    return {"assessment_categories": categories,
            "table_headers": record["headers"], "table_cells": record["cells"],
            "table_subsection": record["section"]}


def table_context_lookup(xml_bytes):
    """Metadatos derivados del XML fijado, también para índices ya construidos."""
    root = ET.fromstring(xml_bytes)
    return {(table.get("id"), row["row_number"]): row_attributes(row)
            for table in root.findall(".//table-wrap") for row in table_records(table)}


def build_passages(xml_bytes, document, counter, *, budget=512):
    root = ET.fromstring(xml_bytes)
    body = root.find("body")
    if body is None:
        raise ValueError("El XML no contiene body JATS.")
    passages, objects = [], []

    def add(content, element_id, sections, kind, *, prefix="", row_number=None):
        parent = fingerprint([document["id"], element_id, row_number, content])
        if counter.count(prefix + content) <= budget:
            pieces = [content]
        else:
            available = budget - counter.count(prefix) - 16
            if available < 32:
                raise ValueError("Encabezados y notas no dejan espacio para la fila.")
            pieces = [content[a:b] for a, b, _ in counter.windows(content, available, 0.15)]
        for part, piece in enumerate(pieces):
            text = prefix + piece
            if counter.count(text) > budget:
                raise ValueError("Pasaje JATS fuera del presupuesto; no se trunca.")
            passages.append({
                "chunk_id": fingerprint([parent, part, text, counter.spec["sha256"]]),
                "parent_chunk_id": parent, "document_id": document["id"],
                "document_version": document["version"], "system": document["system"],
                "document_role": document["role"], "source_url": document["source_url"],
                "source_sha256": document["expected_sha256"],
                "xml_sha256": document["repository_xml_sha256"],
                "xml_element_id": element_id, "table_row": row_number,
                "text": text, "kind": kind, "section_paths": [sections],
                "parent_text": prefix + content if len(pieces) > 1 else None,
                "pages": [], "spans": [], "citation_basis": "JATS section and element ID",
                "token_count_with_passage_prefix": counter.count(text),
                "child_of_oversize_parent": len(pieces) > 1,
                "review_status": "structural_extraction_human_pending",
            })

    def walk(element, sections):
        pending, paragraph_numbers = [], []
        section_prefix = "Section: " + " > ".join(sections) + "\n"

        def flush():
            if pending:
                add("\n\n".join(pending), element.get("id", "body"), sections, "prose",
                    prefix=section_prefix, row_number=None)
                passages[-1]["paragraph_numbers"] = list(paragraph_numbers)
                pending.clear()
                paragraph_numbers.clear()

        for number, child in enumerate(element):
            identifier = child.get("id") or f"{element.get('id', 'body')}-{number}"
            if child.tag != "p":
                flush()
            if child.tag == "sec":
                walk(child, [*sections, text_of(child.find("title"))])
            elif child.tag == "p":
                if text_of(child):
                    candidate = "\n\n".join([*pending, text_of(child)])
                    if pending and counter.count(section_prefix + candidate) > budget:
                        flush()
                    pending.append(text_of(child))
                    paragraph_numbers.append(number)
            elif child.tag == "table-wrap":
                caption = text_of(child.find("caption"))
                notes = text_of(child.find("table-wrap-foot"))
                records = table_records(child)
                objects.append({"id": identifier, "kind": "table", "caption": caption,
                                "notes": notes, "rows": records,
                                "xml": ET.tostring(child, encoding="unicode")})
                for row in records:
                    # Repeat column meanings and footnotes for every retrievable row.
                    prefix = (f"{text_of(child.find('label'))}: {caption}\n"
                              f"Subsection: {row['section']}\nNotes: {notes}\n")
                    content = json.dumps(dict(zip(
                        [f"{j + 1}. {label}" for j, label in enumerate(row["headers"])],
                        row["cells"], strict=True)), ensure_ascii=False)
                    before = len(passages)
                    add(content, identifier, sections, "table_row", prefix=prefix,
                        row_number=row["row_number"])
                    for passage in passages[before:]:
                        passage.update(row_attributes(row))
                        passage["embedding_text"] = passage["text"].replace(
                            f"Notes: {notes}\n", "")
                if notes:
                    add(notes, identifier, sections, "table_notes",
                        prefix=f"{text_of(child.find('label'))}: {caption}\n")
            elif child.tag == "fig":
                objects.append({"id": identifier, "kind": "figure",
                                "caption": text_of(child.find("caption")),
                                "status": "not_interpreted_not_indexed"})
        flush()
    walk(body, [])
    floats = root.find("floats-group")
    if floats is not None:
        walk(floats, ["Tables and figures"])
    if not passages:
        raise ValueError("El documento no produjo pasajes.")
    return passages, objects
