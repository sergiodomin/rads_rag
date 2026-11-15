"""Reranking local por par consulta/pasaje; selección con procedencia conservada."""

import gc
import json
import math
import time

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.paths import within

CATALOG = "data/catalog/reranker_models.v1.json"
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer can only be "
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def catalog(root):
    path = root / CATALOG
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Catálogo de reranking no compatible.")
    return data["models"]


def chat_config(root):
    path = root / "data/catalog/chat_retrieval.v1.json"
    if not path.exists():
        return {"reranker": "none", "candidate_count": 20, "context_policy": "parent"}
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"none", *(s["id"] for s in catalog(root))}
    if (data.get("schema_version") != 1 or not isinstance(data.get("reranker"), str)
            or data["reranker"] not in allowed or type(data.get("candidate_count")) is not int
            or data["candidate_count"] not in {20, 40}
            or data.get("context_policy") not in ("parent", "chunk")):
        raise ValueError("Configuración de recuperación del chat no válida.")
    return {k: data[k] for k in ("reranker", "candidate_count", "context_policy")}


def model_spec(root, identifier, *, verify=False):
    spec = next((s for s in catalog(root) if s["id"] == identifier), None)
    if spec is None:
        raise ValueError("Reranker desconocido.")
    folder = within(root, spec["local_path"])
    for item in spec["files"]:
        path = within(folder, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Prepara primero el reranker: scripts/prepare_reranker.py.")
        if verify and sha256_file(path) != item["sha256"]:
            raise ValueError("Un archivo del reranker ha cambiado.")
    return spec


def pair_tokens(tokenizer, instruction, query, document, limit):
    """Mantiene consulta y plantilla completas; recorta sólo el documento y lo registra."""
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    header = tokenizer.encode(
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ", add_special_tokens=False
    )
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    available = limit - len(prefix) - len(suffix) - len(header)
    if available < 1:
        raise ValueError("La consulta y plantilla superan el presupuesto del reranker.")
    kept = document_ids[:available]
    ids = prefix + header + kept + suffix
    return ids, {
        "pair_tokens": len(ids),
        "original_document_tokens": len(document_ids),
        "scored_document_tokens": len(kept),
        "document_truncated": len(kept) < len(document_ids),
    }


def order_by_scores(sources, scores):
    if len(sources) != len(scores) or any(not math.isfinite(v) for v in scores):
        raise ValueError("Puntuaciones del reranker no válidas.")
    return sorted(range(len(sources)), key=lambda i: (-scores[i], i))


def select_context(sources, *, comparison=False, protected_ids=(), category_first=False, k=4):
    """Deduplica padres y conserva dos plazas por sistema en comparaciones."""
    protected = set(protected_ids)
    order = sorted(
        range(len(sources)),
        key=lambda i: (
            sources[i]["chunk_id"] not in protected,
            not sources[i].get("exact_category_match", False) if category_first else False,
            i,
        ),
    )
    selected, seen, counts = [], set(), {}
    for i in order:
        source = sources[i]
        system = source["system"]
        parent = source["document_id"], source.get("parent_chunk_id", source["chunk_id"])
        if parent in seen or (comparison and counts.get(system, 0) >= k // 2):
            continue
        selected.append(source)
        seen.add(parent)
        counts[system] = counts.get(system, 0) + 1
        if len(selected) == k:
            break
    return [{**s, "id": f"S{i}"} for i, s in enumerate(selected, 1)]


class LocalReranker:
    def __init__(self, root, identifier):
        self.root, self.identifier = root, identifier

    def score(self, query, sources):
        """Un modelo, CUDA float16, batch 1, sin generación ni caché KV persistente."""
        import psutil
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        spec = model_spec(self.root, self.identifier, verify=True)
        if not torch.cuda.is_available():
            raise ValueError("El reranker necesita CUDA en esta configuración.")
        if psutil.virtual_memory().available < 2 * 1024**3:
            raise ValueError("Hay menos de 2 GiB de RAM libre para el reranker.")
        if torch.cuda.mem_get_info(0)[0] < 2 * 1024**3:
            raise ValueError("Libera al menos 2 GiB de VRAM antes del reranking.")
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats(0)
        folder = within(self.root, spec["local_path"])
        model, encoded, output = None, None, None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                folder, local_files_only=True, trust_remote_code=False
            )
            yes = tokenizer.encode("yes", add_special_tokens=False)
            no = tokenizer.encode("no", add_special_tokens=False)
            if len(yes) != 1 or len(no) != 1 or yes == no:
                raise ValueError("El tokenizer no permite el scoring yes/no esperado.")
            model = (
                AutoModelForCausalLM.from_pretrained(
                    folder,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=torch.float16,
                    attn_implementation="sdpa",
                )
                .to("cuda:0")
                .eval()
            )
            torch.cuda.synchronize(0)
            loaded = time.perf_counter()
            scores, details = [], []
            with torch.inference_mode():
                for source in sources:
                    ids, info = pair_tokens(
                        tokenizer,
                        spec["instruction"],
                        query,
                        source["text"],
                        spec["max_pair_tokens"],
                    )
                    encoded = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                    output = model(
                        input_ids=encoded,
                        attention_mask=torch.ones_like(encoded),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    logits = output.logits[0, -1, [no[0], yes[0]]].float()
                    score = torch.softmax(logits, dim=0)[1].item()
                    scores.append(score)
                    details.append(info)
                    encoded, output, logits = None, None, None
            torch.cuda.synchronize(0)
            metrics = {
                "model": self.identifier,
                "revision": spec["revision"],
                "load_and_verify_seconds": loaded - started,
                "scoring_seconds": time.perf_counter() - loaded,
                "pairs": len(sources),
                "batch_size": 1,
                "max_pair_tokens": spec["max_pair_tokens"],
                "dtype": "float16",
                "device": "cuda:0",
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "process_rss_bytes": psutil.Process().memory_info().rss,
                "truncated_documents": sum(d["document_truncated"] for d in details),
            }
            return scores, details, metrics
        finally:
            model, encoded, output = None, None, None
            gc.collect()
            torch.cuda.empty_cache()
