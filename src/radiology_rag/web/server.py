"""Servidor HTTP local con rutas explícitas; no expone el directorio del proyecto."""
import json
import os
import shutil
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from radiology_rag.chat import BusyError, ChatEngine
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab import EvaluationLab
from radiology_rag.generation.local_runtime import ensure_local_runtime
from radiology_rag.generation.ollama import OllamaError
from radiology_rag.generation.vision import VisualRAG
from radiology_rag.paths import within
from radiology_rag.routing import UnifiedChat

ASSETS = Path(__file__).parent / "static"


def build_server(engine, port=8765, lab=None):
    lab = lab or EvaluationLab(engine)
    visual = VisualRAG(engine)
    chat = UnifiedChat(engine, visual)
    class Handler(BaseHTTPRequestHandler):
        server_version = "RadiologyLab"

        def log_message(self, *_args):
            # No guardar preguntas, historial, respuestas ni parámetros de URL.
            pass

        def _allowed(self):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                self._json(421, {"error": "Host no permitido."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in hosts}:
                self._json(403, {"error": "Origen no permitido."})
                return False
            return True

        def _headers(self, code, content_type, length):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                             "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()

        def _json(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._headers(code, "application/json; charset=utf-8", len(data))
            self.wfile.write(data)

        def do_GET(self):
            try:
                if not self._allowed():
                    return
                path = urlsplit(self.path).path
                if path == "/api/status":
                    common = (visual.spec()["model"] if (engine.root /
                              "data/catalog/vision_generator.v1.json").exists() else None)
                    self._json(200, {**engine.status(), "visual": visual.metadata(),
                                     "common_generator": common})
                    return
                if path == "/api/visual":
                    self._json(200, visual.metadata())
                    return
                if path.startswith("/visual/") and path.endswith(".png"):
                    relative = path.removeprefix("/visual/")
                    full_page = relative.endswith("/page.png")
                    identifier = (relative.removesuffix("/page.png") if full_page
                                  else relative.removesuffix(".png"))
                    file = visual.image_path(identifier, full_page=full_page)
                    self._headers(200, "image/png", file.stat().st_size)
                    with file.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile)
                    return
                lab_routes = {"/api/lab": lab.metadata, "/api/lab/state": lab.state,
                              "/api/lab/runs": lab.list_runs}
                if path in lab_routes:
                    self._json(200, lab_routes[path]())
                    return
                if path.startswith("/api/lab/runs/"):
                    self._json(200, lab.read_run(path.removeprefix("/api/lab/runs/")))
                    return
                files = {"/": (ASSETS / "index.html", "text/html; charset=utf-8"),
                         "/lab": (ASSETS / "lab.html", "text/html; charset=utf-8"),
                         "/lab.js": (ASSETS / "lab.js", "text/javascript; charset=utf-8"),
                         "/lab.css": (ASSETS / "lab.css", "text/css; charset=utf-8"),
                         "/app.css": (ASSETS / "app.css", "text/css; charset=utf-8"),
                         "/app.js": (ASSETS / "app.js", "text/javascript; charset=utf-8"),
                         "/document.pdf": (within(engine.root, engine.document["local_path"]),
                                            "application/pdf")}
                for identifier, document in getattr(engine, "documents", {}).items():
                    files[f"/documents/{identifier}.pdf"] = (
                        within(engine.root, document["local_path"]), "application/pdf"
                    )
                if path not in files:
                    self._json(404, {"error": "Ruta no encontrada."})
                    return
                file, content_type = files[path]
                self._headers(200, content_type, file.stat().st_size)
                with file.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ValueError as error:
                self._json(400, {"error": str(error)})
            except FileNotFoundError:
                self._json(404, {"error": "Recurso local no encontrado."})

        def do_POST(self):
            try:
                # Consumir el cuerpo acotado antes de rechazar la petición evita
                # que Windows cierre con RST y se pierda la respuesta HTTP de error.
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    self._json(413, {"error": "La consulta es demasiado grande o está vacía."})
                    return
                self.connection.settimeout(15)
                raw = self.rfile.read(length)
                if not self._allowed():
                    return
                path = urlsplit(self.path).path
                lab_actions = {"/api/lab/start": lab.start,
                               "/api/lab/cancel": lambda _: lab.cancel(),
                               "/api/lab/reference-review": lab.review_case,
                               "/api/lab/result-review": lab.review_record}
                if path not in {"/api/chat", "/api/search", "/api/visual/chat", *lab_actions}:
                    self._json(404, {"error": "Ruta no encontrada."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self._json(415, {"error": "Se requiere application/json."})
                    return
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Se requiere un objeto JSON.")
                result = (visual.respond(payload) if path == "/api/visual/chat" else
                          lab_actions[path](payload) if path in lab_actions
                          else engine.respond(payload, search_only=True) if path == "/api/search"
                          else chat.respond(payload))
                self._json(200, result)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BusyError as error:
                self._json(409, {"error": str(error)})
            except (ValueError, UnicodeError) as error:
                message = ("JSON de consulta no válido." if isinstance(error, json.JSONDecodeError)
                           else str(error))
                self._json(400, {"error": message})
            except OllamaError as error:
                self._json(402 if error.status_code == 402 else 502,
                           {"error": engine.provider_error or str(error)})
            except Exception:
                self._json(500, {"error": "No se pudo completar la consulta. "
                                 "Comprueba el entorno local y la conexión con Ollama."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve(root, *, port=8765, open_browser=False):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    settings = Settings.from_env(root / ".env")
    ensure_local_runtime(root, settings)
    engine = ChatEngine(root, settings)
    engine.refresh_models()
    server = build_server(engine, port)
    address = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Radiology RAG Chat: {address}", flush=True)
    provider = "Ollama local" if settings.is_local else "Ollama Cloud"
    print(f"Embeddings locales en RTX · Generación en {provider} · Ctrl+C para cerrar", flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
