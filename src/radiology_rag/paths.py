from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Funciona desde la raíz o notebooks/, sin depender de una ruta del autor."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data/catalog").is_dir():
            return candidate
    raise FileNotFoundError("Ejecuta desde el repositorio o un subdirectorio suyo.")


def within(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("La ruta sale del directorio permitido.")
    return target
