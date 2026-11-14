"""Staging de artefactos con los permisos normales de su carpeta de destino."""
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def artifact_stage(parent: Path, *, prefix=".stage-"):
    # TemporaryDirectory usa modo 0700: en Windows crea una DACL privada que
    # persiste al renombrar, impidiendo leer al usuario fuera del sandbox.
    parent = parent.resolve()
    stage = parent / f"{prefix}{uuid4().hex}"
    if stage.parent != parent:
        raise ValueError("Staging fuera de la carpeta de destino.")
    stage.mkdir(mode=0o777)
    try:
        yield stage
    finally:
        if stage.exists():
            if stage.resolve().parent != parent:
                raise ValueError("Limpieza de staging fuera de la carpeta de destino.")
            shutil.rmtree(stage)
