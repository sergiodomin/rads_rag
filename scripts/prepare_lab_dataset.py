"""Preparar referencias locales en borrador sin cargar modelos ni modificar revisiones."""

import os
from collections import Counter
from pathlib import Path

from radiology_rag.chat import ChatEngine
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab_dataset import DATASET, prepare_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    engine = ChatEngine(root, Settings.from_env(root / ".env"))
    data = prepare_dataset(engine)
    print(f"Dataset: {DATASET}; revision {data['revision']}")
    splits = dict(Counter(c["split"] for c in data["cases"]))
    print(f"Casos: {len(data['cases'])}; particiones: {splits}")
    print("Referencias extractivas en borrador: requieren revision humana.")


if __name__ == "__main__":
    main()
