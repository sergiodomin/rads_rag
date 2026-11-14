import json

from scripts.check_publish import audit_file


def test_publication_rejects_private_files_and_actual_configured_values():
    assert "private_path" in audit_file("data/evaluations/private.json", b"{}")
    assert "private_path" in audit_file("docs/manual.pdf", b"%PDF")
    assert "private_path" in audit_file(".env", b"KEY=value")
    assert audit_file(".env.example", b"OLLAMA_API_KEY=\n") == []
    assert "configured_secret" in audit_file("README.md", b"sensitive-value", ["sensitive-value"])
    token = "ghp_" + "A" * 36
    assert "github_token" in audit_file("source.py", token.encode())


def test_publication_checks_notebook_outputs_and_size():
    nb = {"cells": [{"cell_type": "code", "outputs": [{"text": "private"}],
                     "execution_count": 1}]}
    assert "notebook_contains_outputs" in audit_file(
        "notebooks/test.ipynb", json.dumps(nb).encode()
    )
    assert "file_larger_than_1MiB" in audit_file("large.txt", b"a" * (1024**2 + 1))
