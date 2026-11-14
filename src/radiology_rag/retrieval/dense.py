"""Operaciones de recuperación visibles también en el notebook 03."""
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from radiology_rag.artifacts import artifact_stage
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.chunking import make_chunk
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import within


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Se requiere una matriz 2D con valores finitos.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Un vector nulo no tiene similitud coseno definida.")
    return matrix / norms


def exact_search(query_vectors, passage_vectors, k=5):
    if k < 1:
        raise ValueError("k debe ser positivo.")
    queries, passages = normalize_rows(query_vectors), normalize_rows(passage_vectors)
    if queries.shape[1] != passages.shape[1] or not len(passages):
        raise ValueError("Las dimensiones o la colección de pasajes no son válidas.")
    scores = queries @ passages.T
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :min(k, len(passages))]
    return indices, np.take_along_axis(scores, indices, axis=1)


def prepare_passages(chunks, text, blocks, counter, *, budget=512, overlap=0.15):
    """Crea hijos trazables de padres largos y conserva todo su texto con offsets."""
    passages = []
    for chunk in chunks:
        parent_id = chunk["chunk_id"]
        if counter.count(chunk["text"]) <= budget:
            passages.append({**chunk, "parent_chunk_id": parent_id,
                             "child_of_oversize_parent": False})
            continue
        for left, right, _ in counter.windows(chunk["text"], budget, overlap):
            child = make_chunk(
                text, blocks, counter, chunk["start"] + left, chunk["start"] + right,
                "embedding_child_tokens",
                {"budget": budget, "overlap": overlap, "parent_chunk_id": parent_id},
            )
            passages.append({**child, "parent_chunk_id": parent_id,
                             "child_of_oversize_parent": True,
                             "review_status": "mechanical_table_child_human_pending"})
    if len({p["chunk_id"] for p in passages}) != len(passages):
        raise ValueError("Hay IDs repetidos en los pasajes preparados.")
    return passages


@dataclass
class LocalEncoder:
    backbone: object
    tokenizer: object
    spec: dict
    device: str = "cpu"

    def encode(self, texts, *, batch_size=1):
        import torch

        if not texts or batch_size < 1:
            raise ValueError("Se necesitan textos y un batch_size positivo.")
        vectors = []
        self.backbone.eval()
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start:start + batch_size], padding=True, truncation=False,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            if batch["input_ids"].shape[1] > self.spec["encoder_max_tokens"]:
                raise ValueError("El batch supera la ventana del encoder.")
            with torch.inference_mode():
                hidden = self.backbone(**batch).last_hidden_state.float()
                mask = batch["attention_mask"]
                if self.spec["pooling"] == "mean":
                    pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                else:
                    # Última posición válida tanto con padding izquierdo como derecho.
                    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
                    positions = positions.expand_as(mask)
                    last = positions.masked_fill(mask == 0, -1).max(1).values
                    pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normalized.cpu().numpy())
            if start % (16 * batch_size) == 0 or start + batch_size >= len(texts):
                print(f"{self.spec['id']}: {min(start + batch_size, len(texts))}/{len(texts)}",
                      flush=True)
        return np.concatenate(vectors).astype(np.float32)


def load_local_encoder(root, spec, *, device="cpu", dtype=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitado, pero PyTorch no puede acceder a la GPU.")
        device = torch.device("cuda", device.index or 0)
    effective_dtype = dtype or spec["dtype"]
    if (device.type == "cuda" and effective_dtype == "bfloat16"
            and not torch.cuda.is_bf16_supported()):
        effective_dtype = "float16"

    path = within(root, spec["local_path"])
    manifest_path = path / ".source.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Preparar modelos: python scripts/prepare_embedding_models.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["revision"] != spec["revision"]
            or manifest["repository"] != spec["repository"]):
        raise ValueError("El modelo local no corresponde a la revisión catalogada.")
    declared = {f["path"]: f for f in spec["files"]}
    if {f["path"] for f in manifest["files"]} != set(declared):
        raise ValueError("El inventario del modelo local está incompleto.")
    for item in manifest["files"]:
        file = within(path, item["path"])
        digest = sha256_file(file)
        pinned = declared[item["path"]]
        if (digest != item["sha256"] or file.stat().st_size != pinned["bytes"]
                or (pinned.get("sha256") and pinned["sha256"] != digest)):
            raise ValueError(f"Modelo local modificado: {item['path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        padding_side="left" if spec["pooling"] == "last_token" else "right",
    )
    backbone = AutoModel.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, effective_dtype),
    )
    if backbone.config.hidden_size != spec["dimension"]:
        raise ValueError("La dimensión cargada difiere del catálogo.")
    backbone.to(device)
    return LocalEncoder(backbone, tokenizer, spec, str(device))


def encode_cached(root, model, spec, records, kind, *, batch_size=1):
    """Guarda matrices normalizadas, orden de filas, tokens y mediciones de inferencia."""
    import torch

    if kind not in {"passage", "query", "query_plain"}:
        raise ValueError("Tipo de entrada desconocido.")
    prefix = {"passage": spec["passage_prefix"], "query": spec["query_prefix"],
              "query_plain": ""}[kind]
    inputs = [prefix + r["text"] for r in records]
    identity = {
        "model": spec, "kind": kind, "prefix": prefix,
        "records": [(r["id"], fingerprint(r["text"])) for r in records],
        "model_manifest_sha256": sha256_file(within(root, spec["local_path"]) / ".source.json"),
        "environment": {p: version(p) for p in ("torch", "transformers", "numpy")},
        "implementation_sha256": sha256_file(Path(__file__)), "batch_size": batch_size,
        "threads": torch.get_num_threads(), "normalize": True, "device": model.device,
        "effective_dtype": str(next(model.backbone.parameters()).dtype),
        "cuda_runtime": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(model.device)
                if model.device.startswith("cuda") else None),
    }
    cache_id = fingerprint(identity)[:20]
    cache_root = root / "data/indices/embedding_cache" / spec["id"]
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / cache_id
    if destination.exists():
        report = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if sha256_file(destination / "vectors.npy") != report["vectors_sha256"]:
            raise ValueError("La matriz de embeddings almacenada fue modificada.")
        vectors = np.load(destination / "vectors.npy", allow_pickle=False)
        if vectors.shape != (len(records), spec["dimension"]):
            raise ValueError("La matriz no corresponde a las filas solicitadas.")
        if not np.allclose(normalize_rows(vectors), vectors, atol=2e-5):
            raise ValueError("Los embeddings almacenados no están normalizados.")
        return vectors, {**report, "cache_hit": True, "cache_path": str(destination)}
    lock = cache_root / f"{cache_id}.lock"
    with lock.open("x", encoding="utf-8") as lock_stream:
        try:
            lengths = [len(ids) for ids in model.tokenizer(
                inputs, add_special_tokens=True, truncation=False, padding=False
            )["input_ids"]]
            if not lengths or max(lengths) > spec["encoder_max_tokens"]:
                raise ValueError("Entrada vacía o mayor que la ventana del encoder.")
            if model.device.startswith("cuda"):
                torch.cuda.synchronize(model.device)
                torch.cuda.reset_peak_memory_stats(model.device)
            started = time.perf_counter()
            vectors = model.encode(inputs, batch_size=batch_size)
            if model.device.startswith("cuda"):
                torch.cuda.synchronize(model.device)
            seconds = time.perf_counter() - started
            vectors = normalize_rows(vectors)
            if vectors.shape != (len(records), spec["dimension"]):
                raise ValueError("Dimensión de embeddings inesperada.")
            with artifact_stage(cache_root, prefix=".encoding-") as directory:
                stage = Path(directory).resolve()
                if not stage.is_relative_to((root / "data/indices").resolve()):
                    raise ValueError("Staging de embeddings fuera del proyecto.")
                np.save(stage / "vectors.npy", vectors, allow_pickle=False)
                report = {
                    **identity, "cache_id": cache_id, "created_at": datetime.now(UTC).isoformat(),
                    "shape": list(vectors.shape), "token_lengths": lengths,
                    "inference_seconds": seconds, "vectors_bytes": vectors.nbytes,
                    "vectors_sha256": sha256_file(stage / "vectors.npy"),
                    "peak_working_set_bytes": _peak_memory(),
                    "peak_vram_allocated_bytes": (
                        torch.cuda.max_memory_allocated(model.device)
                        if model.device.startswith("cuda") else 0
                    ),
                }
                (stage / "manifest.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                stage.rename(destination)
            return vectors, {**report, "cache_hit": False, "cache_path": str(destination)}
        finally:
            lock_stream.close()
            lock.unlink(missing_ok=True)


def _peak_memory():
    import psutil

    memory = psutil.Process().memory_info()
    return getattr(memory, "peak_wset", memory.rss)
