"""Extracción por regiones explícitas: conserva geometría y límites de lo observado."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from radiology_rag.ingestion.catalog import sha256_file


def fingerprint(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def paragraph_lines(lines: list[dict], gap: float = 5) -> list[list[dict]]:
    """Heurística visible: espacios verticales y títulos en negrita; no semántica."""
    groups, current = [], []
    previous_heading = False
    for line in lines:
        chars = line.get("chars", [])
        bold = sum("bold" in c.get("fontname", "").lower() for c in chars)
        heading = bool(chars) and bold > len(chars) * 0.7 and len(line["text"]) < 100
        new_paragraph = bool(current) and (
            heading or line["top"] - current[-1]["bottom"] > gap
        )
        if new_paragraph and not previous_heading:
            groups.append(current)
            current = []
        current.append(line)
        previous_heading = heading
    if current:
        groups.append(current)
    return groups


def join_dropcap(lines: list[dict]) -> list[dict]:
    """Reúne una capitular y la línea que solapa verticalmente, según el mapa revisado."""
    result = []
    index = 0
    while index < len(lines):
        current = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else None
        if (following and len(current["text"]) == 1 and current["text"].isupper()
                and following["text"][:1].islower() and current["bottom"] > following["top"]):
            result.append({
                **following, "text": current["text"] + following["text"],
                "x0": min(current["x0"], following["x0"]),
                "top": min(current["top"], following["top"]),
                "x1": max(current["x1"], following["x1"]),
                "bottom": max(current["bottom"], following["bottom"]),
                "chars": current.get("chars", []) + following.get("chars", []),
            })
            index += 2
        else:
            result.append(current)
            index += 1
    return result


def parse_regions(pdf_path: Path, document: dict, layout: dict, *, x_tolerance: float = 1):
    digest = sha256_file(pdf_path)
    if digest != layout["source_sha256"] or document["id"] != layout["document_id"]:
        raise ValueError("El mapa de regiones no corresponde al documento y hash actuales.")
    reader = PdfReader(pdf_path)
    labels = reader.page_labels
    parser_spec = {
        "name": "pdfplumber.regions", "version": version("pdfplumber"),
        "x_tolerance": x_tolerance, "paragraph_gap_pt": 5,
        "layout_sha256": fingerprint(layout),
    }
    parser_id = fingerprint(parser_spec)
    blocks, images, page_audit = [], [], []
    with pdfplumber.open(pdf_path) as pdf:
        numbers = sorted(p["page_number"] for p in layout["pages"])
        if numbers != list(range(1, len(pdf.pages) + 1)):
            raise ValueError("El mapa debe declarar todas las páginas del PDF.")
        ids = [r["id"] for p in layout["pages"] for r in p["regions"]]
        if len(set(ids)) != len(ids):
            raise ValueError("El mapa contiene identificadores de región duplicados.")
        for page_spec in layout["pages"]:
            number = page_spec["page_number"]
            page = pdf.pages[number - 1]
            page_label = page_spec.get("page_label", labels[number - 1])
            rectangles = [r["bbox"] for r in page_spec["regions"]]
            words = page.extract_words(x_tolerance=x_tolerance)
            unmatched = []
            for word in words:
                x, y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
                if not any(a <= x <= c and b <= y <= d for a, b, c, d in rectangles):
                    unmatched.append({"text": word["text"], "x": x, "y": y})
            audit = {
                "page_number": number, "page_label": page_label,
                "total_words": len(words), "words_outside_regions": unmatched,
                "note": "Fuera del mapa: cabeceras, pies, afiliaciones u omisiones; revisar.",
                "decorative_images": [],
            }
            for i, image in enumerate(page.images, 1):
                obj = {
                    "object_id": f"p{number:02d}-image-{i}", "page_number": number,
                    "page_label": page_label,
                    "bbox": [float(image[k]) for k in ("x0", "top", "x1", "bottom")],
                    "object_type": "raster_image", "status": "not_transcribed_no_ocr",
                    "source_sha256": digest,
                }
                x = (image["x0"] + image["x1"]) / 2
                y = (image["top"] + image["bottom"]) / 2
                decorative = page_spec.get("image_policy") == "decorative_page" or any(
                    a <= x <= c and b <= y <= d
                    for a, b, c, d in layout.get("decorative_image_bands", [])
                )
                if decorative:
                    audit["decorative_images"].append({**obj, "status": "decorative_excluded"})
                else:
                    images.append(obj)
            page_audit.append(audit)
            for region in page_spec["regions"]:
                lines = page.crop(tuple(region["bbox"])).extract_text_lines(
                    x_tolerance=x_tolerance, return_chars=True,
                )
                if region.get("join_dropcap"):
                    lines = join_dropcap(lines)
                groups = [lines] if region["kind"] == "table" else paragraph_lines(lines)
                for index, group in enumerate(groups):
                    if not group:
                        continue
                    text = "\n".join(line["text"] for line in group).strip()
                    if not text:
                        continue
                    bbox = [min(line["x0"] for line in group),
                            min(line["top"] for line in group),
                            max(line["x1"] for line in group),
                            max(line["bottom"] for line in group)]
                    identity = [digest, parser_id, region["id"], index, text]
                    blocks.append({
                        "block_id": fingerprint(identity), "region_id": region["id"],
                        "document_id": document["id"], "document_version": document["version"],
                        "document_role": document["role"], "source_url": document["source_url"],
                        "source_sha256": digest, "page_number": number,
                        "page_label": page_label, "bbox": bbox,
                        "section_path": region["section_path"], "kind": region["kind"],
                        "group_id": region.get("group_id"), "text": text,
                        "unresolved_objects": region.get("unresolved_objects", []),
                        "review_status": layout["review_status"],
                        "parser": parser_spec, "parser_fingerprint": parser_id,
                    })
    images.extend({**item, "source_sha256": digest} for item in layout.get("manual_objects", []))
    return blocks, images, page_audit
