"""Descarga el reranker público fijado en el catálogo; no ejecuta inferencia."""
import json
from pathlib import Path

from prepare_embedding_models import prepare


def main():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "data/catalog/reranker_models.v1.json").read_text())
    for spec in catalog["models"]:
        prepare(root, spec)


if __name__ == "__main__":
    main()
