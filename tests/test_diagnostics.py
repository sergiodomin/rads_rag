import json

import httpx

from scripts.check_ollama import safe_body


def test_diagnostic_masks_raw_and_json_escaped_key():
    secret = 'unit-"quoted\\key\nsecret'
    response = httpx.Response(402, json={"error": f"Bearer {secret}", "key": secret})
    result = safe_body(response, secret)
    assert secret not in result and json.dumps(secret)[1:-1] not in result
    assert json.loads(result)["key"] == "[CLAVE OCULTA]"
    assert "[CLAVE OCULTA]" in safe_body(httpx.Response(402, text=secret), secret)
