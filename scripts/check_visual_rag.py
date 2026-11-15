"""Control pareado texto/imagen sobre el mapa sectorial, con el mismo VLM local."""

import json
import os
from pathlib import Path

from radiology_rag.chat import ChatEngine
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.generation.vision import VisualRAG


def main():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = Path(__file__).resolve().parents[1]
    engine = ChatEngine(root, Settings.from_env(root / ".env"))
    rag = VisualRAG(engine)
    results = []
    for condition in ("text", "image"):
        result = rag.respond(
            {
                "message": "¿De qué color aparece PZ en la leyenda del mapa sectorial?",
                "system": "PI-RADS",
                "visual_asset": "pi-sector-map",
                "visual_input": condition,
            }
        )
        results.append(result)
        print(
            json.dumps(
                {"condition": condition, "answer": result["answer"], "metrics": result["metrics"]},
                ensure_ascii=True,
            ),
            flush=True,
        )
    atomic_json(
        root / "output/visual/paired_smoke.json",
        {
            "results": results,
            "reference_status": "technical_visual_check_only_not_clinically_validated",
        },
    )
    print("Guardado: output/visual/paired_smoke.json", flush=True)


if __name__ == "__main__":
    main()
