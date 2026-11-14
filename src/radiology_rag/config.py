import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://ollama.com"
    model: str = ""
    temperature: float = 0.1
    num_predict: int = 1024
    timeout_seconds: float = 120
    max_retries: int = 2
    max_backoff_seconds: float = 30
    no_cloud: bool = False
    num_ctx: int = 8192
    think: bool | None = None

    @property
    def is_local(self):
        return urlsplit(self.base_url).hostname in {"127.0.0.1", "localhost", "::1"}

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (
            (url.scheme != "https" and not (url.scheme == "http" and self.is_local))
            or not url.netloc or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")
        ):
            raise ValueError("OLLAMA_BASE_URL debe ser un origen HTTPS o HTTP loopback, "
                             "sin /api ni credenciales.")
        if self.no_cloud and not self.is_local:
            raise ValueError("OLLAMA_NO_CLOUD bloquea las llamadas a proveedores remotos.")
        if self.is_local and self.model.endswith((":cloud", "-cloud")):
            raise ValueError("No se admiten modelos Cloud en el servidor local del laboratorio.")
        if not 1024 <= self.num_ctx <= 32768 or self.num_predict >= self.num_ctx:
            raise ValueError("Contexto: 1024..32768; la salida debe ser menor que el contexto.")
        numbers = (self.temperature, self.timeout_seconds, self.max_backoff_seconds)
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError("La configuración numérica debe ser finita.")
        if self.temperature < 0 or self.num_predict < 1 or self.timeout_seconds <= 0:
            raise ValueError("Parámetros de generación o timeout fuera de rango.")
        if not 0 <= self.max_retries <= 5 or not 0 <= self.max_backoff_seconds <= 60:
            raise ValueError("Reintentos: 0..5; backoff máximo: 0..60 segundos.")

    @classmethod
    def from_env(cls, env_file: Path | None = None):
        # Sólo el archivo indicado: no buscar .env en directorios padres ajenos.
        values = {**(dotenv_values(env_file) if env_file else {}), **os.environ}
        think_value = str(values.get("OLLAMA_THINK", "")).lower()
        if think_value not in {"", "true", "false", "1", "0"}:
            raise ValueError("OLLAMA_THINK debe ser true, false o estar vacío.")
        return cls(
            api_key=values.get("OLLAMA_API_KEY", "") or "",
            base_url=values.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/"),
            model=values.get("OLLAMA_MODEL", "") or "",
            temperature=float(values.get("OLLAMA_TEMPERATURE", "0.1")),
            num_predict=int(values.get("OLLAMA_NUM_PREDICT", "1024")),
            timeout_seconds=float(values.get("OLLAMA_TIMEOUT_SECONDS", "120")),
            max_retries=int(values.get("OLLAMA_MAX_RETRIES", "2")),
            max_backoff_seconds=float(values.get("OLLAMA_MAX_BACKOFF_SECONDS", "30")),
            no_cloud=str(values.get("OLLAMA_NO_CLOUD", "0")).lower() in {"1", "true", "yes"},
            num_ctx=int(values.get("OLLAMA_CONTEXT_LENGTH", "8192")),
            think=(None if not think_value else think_value in {"true", "1"}),
        )
