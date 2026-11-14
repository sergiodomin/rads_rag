"""Presupuesto previo con el tokenizer del generador Qwen3 local fijado."""
import json
from functools import lru_cache

from tokenizers import Tokenizer

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within


@lru_cache(maxsize=1)
def local_tokenizer(root):
    spec = json.loads((root / "data/catalog/local_generator.v1.json").read_text(encoding="utf-8"))
    tokenizer_spec = spec["tokenizer"]
    folder = within(root, tokenizer_spec["path"])
    for item in tokenizer_spec["files"]:
        if sha256_file(within(folder, item["path"])) != item["sha256"]:
            raise ValueError("El tokenizer del generador local ha cambiado.")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return spec, tokenizer


def context_budget(messages, tokenizer, *, num_ctx, num_predict):
    # ChatML de Qwen3 sin tools ni reasoning. El margen cubre diferencias de
    # serialización entre el template HF y el de Ollama; se contrasta con usage.
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    prompt += "<|im_start|>assistant\n"
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    budget = tokens + 128
    if budget + num_predict > num_ctx:
        raise ValueError("El contexto recuperado y el historial superan la ventana local. "
                         "Inicia una conversación nueva o acorta la pregunta; no se ha truncado.")
    return {"context_tokens_budgeted": budget, "context_margin_tokens": 128,
            "context_size": num_ctx}


def verify_local_context(root, settings, messages):
    spec, tokenizer = local_tokenizer(root)
    if settings.model != spec["model"]:
        raise ValueError("El chat local está preparado sólo para el generador catalogado.")
    return context_budget(messages, tokenizer, num_ctx=settings.num_ctx,
                          num_predict=settings.num_predict)
