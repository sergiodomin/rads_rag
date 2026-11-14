"""Ejecuta cuadernos desde cero, sin modificar sus fuentes ni requerir API key."""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks", nargs="*", help="Nombres de notebooks; por defecto todos los numerados"
    )
    parser.add_argument("--timeout", type=int, default=300, help="Segundos máximos por celda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "output/notebooks"
    out.mkdir(parents=True, exist_ok=True)
    names = args.notebooks or [
        p.name for p in sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    ]
    report = []

    def write_report(status):
        (out / "execution_report.json").write_text(json.dumps({
            "executed_at": datetime.now(UTC).isoformat(), "python": sys.version,
            "status": status, "results": report,
            "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")

    write_report("running")
    # Kernel temporal del intérprete actual: no escribir configuración global del usuario.
    with tempfile.TemporaryDirectory(prefix="radiology-kernel-") as kernel_dir:
        kernel_root = Path(kernel_dir)
        spec_dir = kernel_root / "kernels/radiology-lab"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Radiology Lab temporary", "language": "python",
            "env": {
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                "IPYTHONDIR": str(kernel_root / "ipython"),
                "MPLCONFIGDIR": str(kernel_root / "matplotlib"),
                "JUPYTER_RUNTIME_DIR": str(kernel_root / "runtime"),
            },
        }), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + previous if previous else "")
        try:
            for name in names:
                path = root / "notebooks" / name
                if path.parent.resolve() != (root / "notebooks").resolve():
                    raise ValueError("Usar nombres de archivos dentro de notebooks/.")
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                started = time.perf_counter()
                print(f"RUN {name}", flush=True)

                def cell_started(cell, cell_index):
                    if cell.cell_type == "code":
                        print(f"  CELL {cell_index + 1}: running", flush=True)

                def cell_executed(cell, cell_index, execute_reply):
                    print(f"  CELL {cell_index + 1}: {execute_reply['content']['status']}",
                          flush=True)
                    for output in cell.get("outputs", []):
                        if output.output_type == "stream":
                            print(output.text, end="", flush=True)

                client = NotebookClient(
                    notebook, timeout=args.timeout, kernel_name="radiology-lab",
                    startup_timeout=30,
                    resources={"metadata": {"path": str(root / "notebooks")}},
                    on_cell_start=cell_started, on_cell_executed=cell_executed,
                )
                client.km = client.create_kernel_manager()
                client.km.kernel_spec_manager = KernelSpecManager(
                    kernel_dirs=[str(kernel_root / "kernels")]
                )
                try:
                    client.execute()
                except Exception as error:
                    report.append({"notebook": name, "status": "failed",
                                   "error_type": type(error).__name__})
                    write_report("failed")
                    raise
                finally:
                    nbformat.write(notebook, out / name)
                elapsed = time.perf_counter() - started
                report.append({"notebook": name, "status": "passed", "seconds": round(elapsed, 3),
                               "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "code_cells": sum(c.cell_type == "code" for c in notebook.cells)})
                print(f"PASS {name}: {elapsed:.2f}s", flush=True)
                write_report("running")
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    write_report("passed")


if __name__ == "__main__":
    main()
