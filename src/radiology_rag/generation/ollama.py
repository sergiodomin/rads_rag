import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from radiology_rag.config import Settings

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Errores sin incluir claves, prompts ni cuerpos de respuesta."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    latency_seconds: float
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    server_total_duration_ns: int | None
    done_reason: str | None

    def metrics(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "content"}


class OllamaCloudClient:
    def __init__(self, settings: Settings, *, transport=None, sleep=time.sleep):
        self.settings = settings
        self._sleep = sleep
        headers = ({"Authorization": f"Bearer {settings.api_key}"}
                   if settings.api_key and not settings.is_local else {})
        self._http = httpx.Client(
            base_url=settings.base_url, headers=headers, timeout=settings.timeout_seconds,
            transport=transport, follow_redirects=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._http.close()

    def _delay(self, response: httpx.Response | None, attempt: int) -> float | None:
        cap = self.settings.max_backoff_seconds
        if response is not None and (value := response.headers.get("Retry-After")):
            try:
                seconds = float(value)
            except ValueError:
                try:
                    seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
                except (TypeError, ValueError):
                    seconds = 0
            # No reintentar antes de lo solicitado ni bloquear indefinidamente.
            if not math.isfinite(seconds) or seconds > cap:
                return None
            return max(0, seconds)
        return min(cap, 2**attempt + random.uniform(0, 0.25))

    def _request(self, method: str, path: str, **kwargs) -> tuple[dict[str, Any], int]:
        for attempt in range(self.settings.max_retries + 1):
            response = None
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError:
                if attempt == self.settings.max_retries:
                    raise OllamaError("Fallo de transporte/timeout; reintentos agotados.") from None
            else:
                if response.is_success:
                    try:
                        data = response.json()
                    except ValueError:
                        raise OllamaError("Respuesta de API no válida: se esperaba JSON.") from None
                    if not isinstance(data, dict) or "error" in data:
                        raise OllamaError("La API devolvió un error o un objeto no válido.")
                    return data, attempt + 1
                retryable = response.status_code in {429, 500, 502, 503, 504}
                if not retryable or attempt == self.settings.max_retries:
                    raise OllamaError(
                        f"Ollama HTTP {response.status_code}. Revisar clave, modelo o cuota.",
                        status_code=response.status_code,
                    )
            delay = self._delay(response, attempt)
            if delay is None:
                raise OllamaError(
                    "Retry-After supera el límite de espera; revisar cuota más tarde."
                )
            self._sleep(delay)
        raise AssertionError("Unreachable")

    def list_models(self) -> list[str]:
        data, _ = self._request("GET", "/api/tags")
        try:
            return [item["name"] for item in data["models"]
                    if not self.settings.is_local or
                    (not item.get("remote_host")
                     and not item["name"].endswith((":cloud", "-cloud")))]
        except (KeyError, TypeError):
            raise OllamaError("Lista de modelos con formato inesperado.") from None

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        if not self.settings.is_local and not self.settings.api_key.strip():
            raise ValueError("Falta OLLAMA_API_KEY; no se ha realizado ninguna llamada.")
        if not self.settings.model.strip():
            raise ValueError("Falta OLLAMA_MODEL; selecciona un nombre de /api/tags.")
        started = time.perf_counter()
        payload = {
            "model": self.settings.model, "messages": messages, "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.num_predict,
            },
        }
        if self.settings.is_local:
            payload["options"].update({"num_ctx": self.settings.num_ctx,
                                       "num_thread": 2, "num_batch": 128})
            payload["keep_alive"] = 0
        if self.settings.think is not None:
            payload["think"] = self.settings.think
        data, attempts = self._request("POST", "/api/chat", json=payload)
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if data.get("done") is not True or not isinstance(content, str):
            raise OllamaError("Respuesta incompleta o sin message.content.")
        result = ChatResult(
            content=content, model=data.get("model", self.settings.model),
            latency_seconds=time.perf_counter() - started, attempts=attempts,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            server_total_duration_ns=data.get("total_duration"),
            done_reason=data.get("done_reason"),
        )
        logger.info("ollama_metrics=%s", result.metrics())
        return result
