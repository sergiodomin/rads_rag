"""Prepara el PDF y XML abiertos fijados en el catálogo BI-RADS."""
import hashlib
import json
from urllib.parse import urlsplit

import httpx

from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.download import acquire, validate_pdf
from radiology_rag.paths import project_root, within


def main():
    root = project_root()
    doc = next(d for d in load_catalog(root / "data/catalog/documents.v1.json")["documents"]
               if d["id"] == "birads-v2025-review-2026")
    pdf = within(root, doc["local_path"])
    if not pdf.exists():
        acquire(root, doc)
    validate_pdf(pdf, doc)
    xml = pdf.with_suffix(".xml")
    if not xml.exists():
        url = doc["repository_xml_url"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "pmc-oa-opendata.s3.amazonaws.com":
            raise ValueError("El XML requiere un origen oficial HTTPS.")
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > 10 * 1024**2:
                        raise ValueError("XML mayor del límite de descarga.")
        if hashlib.sha256(data).hexdigest() != doc["repository_xml_sha256"]:
            raise ValueError("Cambió la identidad del XML.")
        temporary = xml.with_suffix(".xml.part")
        temporary.write_bytes(data)
        temporary.replace(xml)
    if sha256_file(xml) != doc["repository_xml_sha256"]:
        raise ValueError("XML local modificado.")
    print(json.dumps({"document_id": doc["id"], "pdf_verified": True, "xml_verified": True}))


if __name__ == "__main__":
    main()
