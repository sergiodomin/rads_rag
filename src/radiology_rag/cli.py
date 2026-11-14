import argparse
import json
from pathlib import Path

import httpx

from radiology_rag.config import Settings
from radiology_rag.generation.ollama import OllamaCloudClient, OllamaError
from radiology_rag.ingestion.catalog import load_catalog, sha256_file
from radiology_rag.ingestion.download import acquire
from radiology_rag.ingestion.local import import_local
from radiology_rag.paths import project_root, within


def main():
    parser = argparse.ArgumentParser(description="Laboratorio PI-RADS: fuentes y Ollama Cloud")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    sub.add_parser("verify")
    download = sub.add_parser("download")
    download.add_argument("document_id", nargs="?", default="esur-2012")
    download.add_argument("--authorization-note", default="")
    local = sub.add_parser("import", help="Incorpora un PDF local y registra su procedencia")
    local.add_argument("document_id")
    local.add_argument("source", type=Path)
    local.add_argument("--authorization-note", default="")
    process = sub.add_parser("process", help="Extrae bloques y genera las colecciones de chunks")
    process.add_argument("document_id", nargs="?")
    sub.add_parser("models")
    sub.add_parser("check-config")
    sub.add_parser("llm-smoke", help="Llamada real mínima, puede consumir cuota")
    chat = sub.add_parser("chat", help="Abre el chat RAG local con el índice del paso 03")
    chat.add_argument("--port", type=int, default=8765)
    chat.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()
    root = project_root()
    catalog = load_catalog(root / "data/catalog/documents.v1.json")
    try:
        if args.command == "catalog":
            for doc in catalog["documents"]:
                print(doc["id"], doc["version"], doc["status"], doc["source_url"])
        elif args.command in {"download", "import"}:
            matches = [d for d in catalog["documents"] if d["id"] == args.document_id]
            if not matches:
                parser.error("Documento no encontrado en el catálogo.")
            if args.command == "import":
                record = import_local(root, matches[0], args.source,
                                      authorization_note=args.authorization_note)
            else:
                record = acquire(root, matches[0], authorization_note=args.authorization_note)
            print(json.dumps(record, indent=2, ensure_ascii=False))
        elif args.command == "process":
            from radiology_rag.ingestion.pipeline import process_document

            report = process_document(root, args.document_id)
            print(json.dumps({key: report[key] for key in (
                "document_id", "run_id", "pages", "blocks_total", "retrieval_blocks",
                "selected_characters", "unresolved_objects", "metrics",
            )}, indent=2, ensure_ascii=False))
        elif args.command == "verify":
            found = 0
            for doc in catalog["documents"]:
                path = within(root, doc["local_path"])
                if not path.exists():
                    print(f"{doc['id']}: no incorporado ({doc['status']})")
                    continue
                manifest = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
                digest = sha256_file(path)
                if digest != manifest["sha256"] or (
                    doc.get("expected_sha256") and digest != doc["expected_sha256"]
                ):
                    raise ValueError(f"Integridad fallida: {doc['id']}")
                if manifest["document_id"] != doc["id"] or manifest["version"] != doc["version"]:
                    raise ValueError(f"Identidad del manifiesto incorrecta: {doc['id']}")
                found += 1
                print(f"{doc['id']}: SHA-256 verificado")
            if not found:
                raise ValueError("No hay documentos locales para verificar.")
        elif args.command == "chat":
            from radiology_rag.web.server import serve

            serve(root, port=args.port, open_browser=args.open_browser)
        else:
            settings = Settings.from_env(root / ".env")
            if args.command == "check-config":
                print(json.dumps({"base_url": settings.base_url, "model": settings.model,
                                  "api_key_present": bool(settings.api_key)}, indent=2))
            else:
                with OllamaCloudClient(settings) as client:
                    if args.command == "models":
                        print("\n".join(client.list_models()))
                    else:
                        result = client.chat([{"role": "user", "content": "Responde sólo: OK"}])
                        print(result.content)
                        print(json.dumps(result.metrics(), indent=2))
    except (OllamaError, httpx.HTTPError, ValueError, OSError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
