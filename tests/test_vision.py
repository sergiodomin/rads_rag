import base64
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

import radiology_rag.generation.vision as module
from radiology_rag.chat import BusyError
from radiology_rag.config import Settings
from radiology_rag.evaluation.lab_dataset import atomic_json
from radiology_rag.generation.ollama import OllamaCloudClient
from radiology_rag.generation.vision import VisualRAG
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.web.server import build_server


@pytest.fixture
def visual(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"synthetic PDF identity")
    folder = tmp_path / "data/processed/multimodal/v1"
    folder.mkdir(parents=True)
    image = folder / "map.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1cAAAAASUVORK5CYII="
        )
    )
    asset = {
        "id": "map",
        "document_id": "unit",
        "document": "Synthetic",
        "system": "PI-RADS",
        "document_version": "1",
        "document_role": "test",
        "source_sha256": sha256_file(pdf),
        "xml_sha256": None,
        "bbox": [0, 0, 10, 10],
        "page_size_points": [100, 100],
        "image_size": [1, 1],
        "image_path": image.relative_to(tmp_path).as_posix(),
        "image_sha256": sha256_file(image),
        "page_image_path": image.relative_to(tmp_path).as_posix(),
        "page_image_sha256": sha256_file(image),
        "page_number": 1,
        "page_label": "1",
        "title": "Map",
        "kind": "figure",
        "keywords": "mapa sectorial PZ colores",
        "caption": "synthetic caption",
        "pdf_extracted_text": "synthetic PDF text",
        "pdf_url": "/documents/unit.pdf#page=1",
    }
    index = {"schema_version": 1, "assets": [asset]}
    index["build_id"] = fingerprint(index)
    atomic_json(folder / "manifest.json", index)
    spec = {
        "schema_version": 1,
        "model": "vision-unit",
        "ollama_digest": "digest-unit",
        "max_images": 1,
        "max_image_edge": 1008,
        "num_ctx": 8192,
        "image_token_reserve": 2048,
        "num_predict": 384,
        "template_token_margin": 256,
        "max_text_characters": 6000,
        "tokenizer": {"path": "tokenizer-unit"},
    }
    atomic_json(tmp_path / "data/catalog/vision_generator.v1.json", spec)
    engine = SimpleNamespace(
        root=tmp_path,
        _phase="ready",
        _lock=threading.Lock(),
        documents={"unit": {"local_path": "manual.pdf", "expected_sha256": asset["source_sha256"]}},
        settings=Settings(
            base_url="http://127.0.0.1:11434", api_key="private-unit-key", model="text-unit"
        ),
        release_encoder=lambda: None,
    )
    return VisualRAG(engine)


@pytest.mark.parametrize("condition", ["image", "text", "combined"])
def test_visual_conditions_send_only_registered_bytes_and_never_persist_base64(
    visual, monkeypatch, condition
):
    requests = []

    def provider(request):
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "vision-unit", "digest": "digest-unit"}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["vision", "thinking"]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "vision-unit" and payload["keep_alive"] == 0
        assert "think" not in payload
        message = payload["messages"][-1]
        if condition != "text":
            assert len(message["images"]) == 1
            assert base64.b64decode(message["images"][0]) == visual.image_path("map").read_bytes()
        else:
            assert "images" not in message
        assert ("synthetic caption" in message["content"]) == (condition != "image")
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": "synthetic answer [S1]"},
            },
        )

    monkeypatch.setattr(
        module,
        "OllamaCloudClient",
        lambda settings: OllamaCloudClient(settings, transport=httpx.MockTransport(provider)),
    )
    monkeypatch.setattr(visual, "budget", lambda *_: {})
    result = visual.respond({"message": "color PZ", "visual_input": condition})
    assert requests and result["answer"] == "synthetic answer [S1]"
    assert result["metrics"]["images_sent"] == int(condition != "text")
    assert not visual.engine._lock.locked() and visual.engine._phase == "ready"
    assert all("images" not in m for m in result["trace"]["messages_without_images"])
    saved = next((visual.root / "output/visual/runs").glob("*.json")).read_text(encoding="utf-8")
    assert "private-unit-key" not in saved and "iVBORw0KGgo" not in saved


def test_visual_selection_filters_sources_and_rejects_arbitrary_ids(visual):
    assert visual.select("mapa sectorial", "all")[0]["id"] == "map"
    with pytest.raises(ValueError):
        visual.select("BI-RADS", "PI-RADS", "map")
    with pytest.raises(ValueError):
        visual.select("mapa", "all", "../../.env")
    with pytest.raises(ValueError):
        visual.image_path("../../.env")


def test_visual_client_forwards_schema_without_altering_image_bytes(visual, monkeypatch):
    schema = {"type": "object", "properties": {"abstain": {"type": "boolean"}}}

    def provider(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "vision-unit", "digest": "digest-unit"}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["vision"]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        data = json.loads(request.content)
        assert data["format"] == schema and data["messages"][-1]["images"] == ["test-bytes"]
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"done": True, "message": {"content": '{"abstain":true}'}})

    monkeypatch.setattr(
        module,
        "OllamaCloudClient",
        lambda settings: OllamaCloudClient(settings, transport=httpx.MockTransport(provider)),
    )
    result = visual.generate(
        visual.spec(),
        [{"role": "user", "content": "test", "images": ["test-bytes"]}],
        response_format=schema,
    )
    assert json.loads(result.content)["abstain"] is True


def test_visual_detects_image_tampering_and_releases_shared_lock(visual, monkeypatch):
    path = visual.image_path("map")
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="recorte"):
        visual.respond({"message": "PZ", "visual_asset": "map"})
    assert not visual.engine._lock.locked()
    with visual.engine._lock, pytest.raises(BusyError):
        visual.respond({"message": "PZ"})


def test_budget_reserves_visual_tokens_and_rejects_overflow(visual):
    spec = visual.spec()
    visual._tokenizer_hash = fingerprint(spec["tokenizer"])
    visual._tokenizer = SimpleNamespace(encode=lambda text, **_: SimpleNamespace(ids=list(text)))
    messages = [{"role": "user", "content": "short question"}]
    a, b = visual.budget(spec, messages, True), visual.budget(spec, messages, False)
    assert a["budgeted_total_tokens"] - b["budgeted_total_tokens"] == 2048
    with pytest.raises(ValueError, match="presupuesto"):
        visual.budget(spec, [{"role": "user", "content": "a" * 8192}], True)


def test_visual_rejects_unregistered_image_path_even_with_valid_manifest_hash(visual):
    data = visual.index()
    data["assets"][0]["image_path"] = "manual.pdf"
    data["build_id"] = fingerprint({k: v for k, v in data.items() if k != "build_id"})
    atomic_json(visual.folder / "manifest.json", data)
    with pytest.raises(ValueError, match="fuera"):
        visual.image_path("map")


def test_visual_png_http_route_exposes_only_registered_assets(visual):
    server = build_server(visual.engine, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            response = client.get("/visual/map.png")
            assert response.status_code == 200 and response.headers["content-type"] == "image/png"
            assert client.get("/visual/map/page.png").status_code == 200
            assert client.get("/visual/unknown.png").status_code == 400
            assert client.get("/api/visual").json()["assets"][0]["id"] == "map"
            assert (
                client.post(
                    "/api/visual/chat",
                    json={"message": "PZ"},
                    headers={"Origin": "https://external.invalid"},
                ).status_code
                == 403
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
