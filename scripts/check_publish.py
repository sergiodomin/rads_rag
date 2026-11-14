"""Audita archivos publicables sin mostrar valores de credenciales ni texto encontrado."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import PurePosixPath

from dotenv import dotenv_values

from radiology_rag.paths import project_root

PRIVATE_DIRS = {".venv", ".uv-cache", "tmp", "output", ".aws", ".ssh", ".codex", ".agents"}
PRIVATE_SUFFIXES = {".pdf", ".gguf", ".safetensors", ".npy", ".npz", ".zip", ".exe",
                    ".dll", ".dmp", ".log", ".pem", ".key", ".p12", ".pfx"}
SECRET_PATTERNS = {
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----"),
    "provider_key": re.compile(r"\b(?:sk-|gsk_)[A-Za-z0-9_-]{24,}"),
    "hardcoded_credential": re.compile(
        r"(?i)(?:api[_-]?key|authorization|password|token|secret)[\"']?\s*[:=]\s*"
        r"[\"'][A-Za-z0-9._/-]{24,}[\"']"
    ),
}


def audit_file(path, content, known_secrets=()):
    file = PurePosixPath(path)
    errors = []
    if (any(part in PRIVATE_DIRS for part in file.parts)
            or file.suffix.lower() in PRIVATE_SUFFIXES
            or (file.name.startswith(".env") and path != ".env.example")
            or (path.startswith("data/") and file.parts[1] in
                {"raw", "processed", "indices", "runs", "evaluations"}
                and file.name != ".gitkeep")):
        errors.append("private_path")
    if len(content) > 1024**2:
        errors.append("file_larger_than_1MiB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [*errors, "non_text_file"]
    if any(value and (value in text or json.dumps(value)[1:-1] in text)
           for value in known_secrets):
        errors.append("configured_secret")
    errors.extend(label for label, pattern in SECRET_PATTERNS.items() if pattern.search(text))
    if file.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(cell.get("outputs") or cell.get("execution_count") is not None
               for cell in notebook["cells"]):
            errors.append("notebook_contains_outputs")
    return errors


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="Revisar blobs del índice Git")
    group.add_argument("--tracked", action="store_true", help="Revisar archivos versionados")
    args = parser.parse_args()
    root = project_root()
    # Sólo comparar valores en memoria; no imprimirlos ni incluirlos en el informe.
    values = dotenv_values(root / ".env") if (root / ".env").exists() else {}
    secrets = [value for key, value in values.items()
               if re.search(r"KEY|TOKEN|SECRET|PASSWORD", key) and value and len(value) >= 8]
    options = ["--cached"] if args.staged or args.tracked else [
        "--cached", "--others", "--exclude-standard",
    ]
    paths = sorted(set(p for p in git(root, "ls-files", "-z", *options).decode().split("\0") if p))
    if not paths:
        raise SystemExit("No hay archivos para revisar.")
    failures, manifest = [], []
    for path in paths:
        content = git(root, "show", f":{path}") if args.staged else (root / path).read_bytes()
        errors = audit_file(path, content, secrets)
        if errors:
            failures.append({"path": path, "checks": errors})
        manifest.append({"path": path, "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest()})
    total = sum(item["bytes"] for item in manifest)
    if total > 2 * 1024**2:
        failures.append({"path": "repository", "checks": ["larger_than_2MiB"]})
    report = {"status": "failed" if failures else "passed", "files": len(paths),
              "bytes": total, "staged": args.staged, "failures": failures, "manifest": manifest}
    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "publication_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "files", "bytes", "failures")},
                     ensure_ascii=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
