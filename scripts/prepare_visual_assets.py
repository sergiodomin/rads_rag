"""Renderiza recortes registrados con procedencia; no interpreta las imágenes."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.jats import text_of
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import within


def prepare(root):
    seed_path = root / "data/catalog/visual_regions.v1.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    documents = {
        d["id"]: d for d in load_catalog(root / "data/catalog/documents.v1.json")["documents"]
    }
    assets = []
    folder = root / "data/processed/multimodal/v1"
    folder.mkdir(parents=True, exist_ok=True)
    for region in seed["regions"]:
        document = documents[region["document_id"]]
        source = within(root, document["local_path"])
        digest = sha256_file(source)
        if digest != document["expected_sha256"]:
            raise ValueError("El PDF no corresponde al catálogo.")
        x0, top, x1, bottom = region["bbox"]
        with pdfplumber.open(source) as pdf:
            page = pdf.pages[region["page_number"] - 1]
            width, height = float(page.width), float(page.height)
            if not 0 <= x0 < x1 <= width or not 0 <= top < bottom <= height:
                raise ValueError("Recorte fuera de la página.")
            baseline = page.within_bbox(tuple(region["bbox"])).extract_text() or ""
        rendered = pdfium.PdfDocument(str(source))
        page = rendered[region["page_number"] - 1]
        try:
            # PDFium crop usa márgenes desde izquierda/abajo/derecha/arriba.
            image = page.render(scale=2, crop=(x0, height - bottom, width - x1, top)).to_pil()
            image.thumbnail((1008, 1008))
            target = folder / (region["id"] + ".png")
            image.save(target)
            pixel_size = list(image.size)
            page_image = page.render(scale=1.5).to_pil()
            page_image.thumbnail((1008, 1008))
            page_target = folder / (region["id"] + ".page.png")
            page_image.save(page_target)
        finally:
            page.close()
            rendered.close()
        caption, xml_digest = "", None
        if region.get("xml_element_id"):
            xml_path = source.with_suffix(".xml")
            xml_digest = sha256_file(xml_path)
            if xml_digest != document["repository_xml_sha256"]:
                raise ValueError("El XML no corresponde al catálogo.")
            element = next(
                e
                for e in ET.parse(xml_path).getroot().iter("fig")
                if e.get("id") == region["xml_element_id"]
            )
            caption = text_of(element.find("caption"))
        assets.append(
            {
                **region,
                "source_sha256": digest,
                "xml_sha256": xml_digest,
                "document": document["title"],
                "system": document["system"],
                "document_version": document["version"],
                "document_role": document["role"],
                "page_size_points": [width, height],
                "image_size": pixel_size,
                "image_path": target.relative_to(root).as_posix(),
                "image_sha256": sha256_file(target),
                "page_image_path": page_target.relative_to(root).as_posix(),
                "page_image_sha256": sha256_file(page_target),
                "caption": caption,
                "pdf_extracted_text": baseline,
                "image_render_scale": 2,
                "max_edge_pixels": 1008,
                "geometry_review": "manual_pdf_region_selection",
                "interpretation_status": "not_interpreted",
                "pdf_url": f"/documents/{document['id']}.pdf#page={region['page_number']}",
            }
        )
    record = {
        "schema_version": 1,
        "seed_sha256": sha256_file(seed_path),
        "coordinate_system": seed["coordinate_system"],
        "assets": assets,
    }
    record["build_id"] = fingerprint(record)
    atomic_json(folder / "manifest.json", record)
    return record


if __name__ == "__main__":
    data = prepare(Path(__file__).resolve().parents[1])
    print(f"Recortes: {len(data['assets'])}; build: {data['build_id']}")
