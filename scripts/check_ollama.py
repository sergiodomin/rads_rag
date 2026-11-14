"""Diagnóstico manual: una petición mínima real, sin reintentos ni mostrar la clave."""
import argparse
import json
import sys
from dataclasses import replace

import httpx

from radiology_rag.config import Settings
from radiology_rag.paths import project_root


def safe_body(response, api_key):
    try:
        body = json.dumps(response.json(), ensure_ascii=True, indent=2)
    except ValueError:
        body = response.text
    if api_key:
        body = body.replace(api_key, "[CLAVE OCULTA]")
        body = body.replace(json.dumps(api_key)[1:-1], "[CLAVE OCULTA]")
    return body[:6000]


def main():
    parser = argparse.ArgumentParser(
        description="Prueba Ollama con una petición real mínima. Cloud puede consumir cuota. "
                    "Muestra errores del proveedor sin mostrar tu clave; no hace reintentos."
    )
    parser.add_argument("--model", help="Modelo opcional; por defecto OLLAMA_MODEL de .env")
    parser.add_argument("--list-only", action="store_true", help="Sólo listar modelos, sin generar")
    args = parser.parse_args()
    settings = Settings.from_env(project_root() / ".env")
    if args.model:
        settings = replace(settings, model=args.model)
    print(f"Proveedor: {settings.base_url}", flush=True)
    print(f"Modelo configurado: {settings.model or '(vacío)'}", flush=True)
    print(f"Clave presente: {'sí' if settings.api_key.strip() else 'no'}", flush=True)
    if not settings.is_local and not settings.api_key.strip():
        print("Falta OLLAMA_API_KEY en .env; no se ha hecho ninguna petición.")
        return 1
    try:
        with httpx.Client(
            base_url=settings.base_url,
            headers=({"Authorization": f"Bearer {settings.api_key}"}
                     if not settings.is_local else {}),
            timeout=settings.timeout_seconds,
            follow_redirects=False,
        ) as client:
            tags = client.get("/api/tags")
            print(f"GET /api/tags -> HTTP {tags.status_code}", flush=True)
            if not tags.is_success:
                print(safe_body(tags, settings.api_key))
                return 1
            data = tags.json()
            models = [item["name"] for item in data.get("models", [])]
            if args.list_only:
                print("\n".join(models))
                return 0
            model = settings.model
            candidate = model.removesuffix("-cloud")
            if model not in models and candidate in models:
                model = candidate
            if not model.strip():
                print("Falta OLLAMA_MODEL. Ejecuta con --list-only y elige uno con --model.")
                return 1
            print(f"Modelo enviado a la API: {model}", flush=True)
            print("POST /api/chat -> petición real mínima; sin documentos ni historial.",
                  flush=True)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Responde solamente: OK"}],
                "stream": False,
                "options": {"temperature": settings.temperature, "num_predict": 128},
            }
            if settings.is_local:
                payload["options"].update({"num_ctx": settings.num_ctx, "num_thread": 2,
                                           "num_batch": 128})
                payload["keep_alive"] = 0
            response = client.post("/api/chat", json=payload)
            print(f"HTTP {response.status_code} {response.reason_phrase}", flush=True)
            print(safe_body(response, settings.api_key), flush=True)
            return 0 if response.is_success else 1
    except httpx.HTTPError as error:
        message = str(error).replace(settings.api_key, "[CLAVE OCULTA]")
        print(f"Fallo de conexión: {type(error).__name__}: {message}", flush=True)
        return 1
    except (ValueError, KeyError, TypeError):
        print("El proveedor devolvió una respuesta con formato inesperado.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
