"""Arranque del runtime portable local, sin instalar servicios ni habilitar Cloud."""
import json
import os
import subprocess
import time

import httpx


def ensure_local_runtime(root, settings):
    if not settings.is_local:
        return
    with httpx.Client(base_url=settings.base_url, timeout=2) as client:
        try:
            if client.get("/api/version").is_success:
                return
        except httpx.HTTPError:
            pass
    executable = root / "tmp/ollama_runtime/0.40.16/ollama.exe"
    if not executable.is_file():
        raise FileNotFoundError("Preparar Ollama portable: python scripts/prepare_local_ollama.py")
    models = root / "data/raw/ollama_models"
    models.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OLLAMA_HOST": settings.base_url.removeprefix("http://"),
           "OLLAMA_MODELS": str(models), "OLLAMA_NO_CLOUD": "1",
           "CUDA_VISIBLE_DEVICES": "0", "OLLAMA_VULKAN": "0",
           "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
           "OLLAMA_CONTEXT_LENGTH": str(settings.num_ctx), "OLLAMA_KEEP_ALIVE": "0"}
    env.pop("OLLAMA_API_KEY", None)
    output = root / "output"
    output.mkdir(exist_ok=True)
    with (output / "ollama_local.stdout.log").open("wb") as stdout, (
        output / "ollama_local.stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(
            [str(executable), "serve"], cwd=root, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    (output / "ollama_local_runtime.json").write_text(json.dumps({
        "pid": process.pid, "url": settings.base_url, "no_cloud": True,
        "models": str(models), "cuda_visible_devices": "0",
    }, indent=2) + "\n", encoding="utf-8")
    print("Preparando Ollama local…", flush=True)
    with httpx.Client(base_url=settings.base_url, timeout=2) as client:
        for _ in range(240):
            if process.poll() is not None:
                raise RuntimeError("Ollama local terminó; revisar output/ollama_local.stderr.log.")
            try:
                if client.get("/api/version").is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
    raise RuntimeError("Ollama local no respondió a tiempo; revisar su registro.")
