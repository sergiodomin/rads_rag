"""Prueba real del chat local con una pregunta documental, sin llamadas Cloud."""
import json
import time
from datetime import UTC, datetime

import httpx

from radiology_rag.paths import project_root


def main():
    root = project_root()
    started = time.perf_counter()
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=300) as client:
        response = client.get("/api/status")
        response.raise_for_status()
        status = response.json()
        assert status["provider_kind"] == "local" and status["generation_ready"], status
        print("Local model:", status["model"], flush=True)
        pilot = json.loads((root / "data/evaluations/pirads-2.1-2019-retrieval-pilot.v1.json")
                           .read_text(encoding="utf-8"))
        question = next(c for c in pilot["cases"] if c["id"] == "index-lesion")["questions"]["es"]
        print("Retrieving and generating locally…", flush=True)
        response = client.post("/api/chat", json={"message": question, "model": status["model"],
                                                "encoder": "qwen3-0.6b"})
        if response.is_error:
            print(response.status_code, response.json(), flush=True)
        response.raise_for_status()
        result = response.json()
        (root / "output/chat_local_example.json").write_text(
            json.dumps({"question": question, **result}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(result["answer"], flush=True)
        assert not result["abstained"] and result["cited_sources"]
        assert not result["truncated"]
        metrics = result["metrics"]
        assert metrics["prompt_tokens"] <= metrics["context_tokens_budgeted"]
        assert metrics["context_tokens_budgeted"] + 512 <= metrics["context_size"]
        with httpx.Client(base_url="http://127.0.0.1:11434", timeout=10) as native:
            for _ in range(20):
                if not native.get("/api/ps").json()["models"]:
                    break
                time.sleep(0.25)
            else:
                raise AssertionError("El generador no se descargó de memoria.")
        assert client.get("/api/status").json()["loaded_encoder"] is None
        report = {"checked_at": datetime.now(UTC).isoformat(), "status": "passed",
                  "model": status["model"], "provider": status["provider"],
                  "seconds": time.perf_counter() - started, "cloud_calls": 0,
                  "cited_sources": result["cited_sources"], "metrics": metrics,
                  "generator_unloaded": True, "encoder_released": True}
        (root / "output/chat_validation_local.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
