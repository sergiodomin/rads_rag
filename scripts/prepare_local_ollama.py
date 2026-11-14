"""Prepara el runtime portable oficial; no instala servicios ni cambia drivers."""
import hashlib
import json
import time
import zipfile
from datetime import UTC, datetime

import httpx

from radiology_rag.paths import project_root

VERSION = "0.40.16"
URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-windows-amd64.zip"
SHA256 = "50fcf5ad54277f2fa5d173ebfc93bcb017adbc0c9ef2314daabc96ca9e6aa64e"
SIZE = 1618244995


def main():
    root = project_root()
    folder = root / f"tmp/ollama_runtime/{VERSION}"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "ollama-windows-amd64.zip"
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        existing = partial.stat().st_size if partial.exists() else 0
        started = time.monotonic()
        with httpx.Client(follow_redirects=True, timeout=60) as client, client.stream(
            "GET", URL, headers={"Range": f"bytes={existing}-"} if existing else {}
        ) as response:
            response.raise_for_status()
            append = bool(existing and response.status_code == 206)
            downloaded = existing if append else 0
            last_notice = downloaded
            with partial.open("ab" if append else "wb") as stream:
                for block in response.iter_bytes(1024 * 1024):
                    stream.write(block)
                    downloaded += len(block)
                    if downloaded > SIZE:
                        raise ValueError("Descarga mayor que el activo oficial.")
                    if downloaded - last_notice >= 64 * 1024**2:
                        print(f"Runtime {downloaded / SIZE:.1%} · {downloaded / 1024**2:.0f} MiB",
                              flush=True)
                        last_notice = downloaded
            if downloaded != SIZE:
                raise ValueError("Descarga incompleta; se puede reanudar ejecutando el script.")
        partial.replace(archive)
        print(f"Downloaded in {time.monotonic() - started:.1f}s", flush=True)
    with archive.open("rb") as archive_stream:
        digest = hashlib.file_digest(archive_stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("SHA-256 del runtime distinto del publicado por GitHub.")
    print("Official ZIP SHA-256 verified", flush=True)
    excluded = []
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            if any(part in item.filename.lower() for part in
                   ("cuda_v13", "vulkan", "rocm", "mlx")):
                excluded.append(item.filename)
                continue
            target = (folder / item.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("Ruta no válida dentro del ZIP.")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as source, target.open("wb") as destination:
                    import shutil

                    shutil.copyfileobj(source, destination, length=1024 * 1024)
    if not (folder / "ollama.exe").is_file():
        raise FileNotFoundError("El paquete no contiene ollama.exe.")
    (folder / "source.json").write_text(json.dumps({
        "version": VERSION, "url": URL, "archive_sha256": digest,
        "prepared_at": datetime.now(UTC).isoformat(),
        "excluded_non_cuda12_backends": excluded,
    }, indent=2) + "\n", encoding="utf-8")
    print("Runtime ready", folder / "ollama.exe", flush=True)


if __name__ == "__main__":
    main()
