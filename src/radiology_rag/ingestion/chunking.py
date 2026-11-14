"""Chunking por tokens reales y por unidades de estructura, sin LLM ni encoder."""
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import within


@dataclass
class TokenCounter:
    tokenizer: Tokenizer
    spec: dict

    @classmethod
    def from_project(cls, root: Path):
        spec = json.loads((root / "data/catalog/tokenizers.v1.json").read_text(encoding="utf-8"))[
            "tokenizers"
        ][0]
        path = within(root, spec["local_path"])
        if not path.is_file():
            raise FileNotFoundError("Falta tokenizer: uv run python scripts/prepare_tokenizer.py")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError("El tokenizer local no coincide con el hash fijado.")
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return cls(tokenizer, spec)

    def count(self, text: str) -> int:
        """Incluye prefijo passage y tokens especiales, sin truncar."""
        return len(self.tokenizer.encode(self.spec["passage_prefix"] + text).ids)

    def windows(self, text: str, budget: int, overlap: float = 0):
        """Offsets sobre el texto original: nunca reconstruirlo decodificando IDs."""
        if budget < 16 or not 0 <= overlap < 1:
            raise ValueError("budget debe ser >=16 y overlap debe estar en [0,1).")
        start = 0
        overhead = self.count("")
        while start < len(text):
            tail = text[start:]
            offsets = self.tokenizer.encode(tail, add_special_tokens=False).offsets
            if not offsets:
                if tail.strip():
                    raise ValueError("Texto no vacío sin tokens: revisar normalización.")
                break
            limit = min(len(offsets), budget - overhead)
            end = len(text) if limit == len(offsets) else start + offsets[limit][0]
            while end > start and self.count(text[start:end]) > budget:
                limit -= 1
                end = start + offsets[limit][0]
            if end <= start:
                raise ValueError("No se puede cortar con ese presupuesto sin perder caracteres.")
            yield start, end, self.count(text[start:end])
            if end == len(text):
                break
            if overlap:
                current = self.tokenizer.encode(text[start:end], add_special_tokens=False).offsets
                keep = min(len(current) - 1, math.floor(budget * overlap))
                next_start = start + current[-keep][0] if keep else end
                start = next_start if next_start > start else end
            else:
                start = end


def flatten_blocks(blocks: list[dict]) -> tuple[str, list[dict]]:
    """Corpus común para todos los chunkers; sólo prosa y tablas, mismo orden."""
    pieces, located, cursor = [], [], 0
    for block in blocks:
        if block["kind"] not in {"prose", "table"}:
            continue
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        located.append({**block, "start": cursor, "end": cursor + len(block["text"])})
        pieces.append(block["text"])
        cursor += len(block["text"])
    return "".join(pieces), located


def make_chunk(text, blocks, counter, start, end, strategy, parameters):
    contributors = [b for b in blocks if b["start"] < end and b["end"] > start]
    if not contributors:
        raise ValueError("Chunk sin procedencia.")
    first = contributors[0]
    spans = [{
        "block_id": b["block_id"], "page_number": b["page_number"],
        "page_label": b["page_label"], "region_id": b["region_id"],
        "block_start": max(start, b["start"]) - b["start"],
        "block_end": min(end, b["end"]) - b["start"],
        "bbox": b["bbox"], "bbox_scope": "whole_block_not_character_slice",
        "section_path": b["section_path"], "kind": b["kind"],
    } for b in contributors]
    chunk_text = text[start:end]
    token_count = counter.count(chunk_text)
    sections = list(dict.fromkeys(tuple(b["section_path"]) for b in contributors))
    corpus_hash = fingerprint([(b["block_id"], b["start"], b["end"]) for b in blocks])
    token_identity = [counter.spec["revision"], counter.spec["sha256"]]
    identity = [first["document_id"], first["document_version"],
                first["source_sha256"], corpus_hash, strategy, parameters,
                token_identity, start, end, chunk_text]
    return {
        "chunk_id": fingerprint(identity), "document_id": first["document_id"],
        "document_version": first["document_version"], "document_role": first["document_role"],
        "source_url": first["source_url"], "source_sha256": first["source_sha256"],
        "parser_fingerprint": first["parser_fingerprint"], "corpus_fingerprint": corpus_hash,
        "strategy": strategy, "parameters": parameters,
        "tokenizer_repository": counter.spec["repository"],
        "tokenizer_revision": counter.spec["revision"], "tokenizer_sha256": counter.spec["sha256"],
        "token_count_with_passage_prefix": token_count,
        "exceeds_encoder_window": token_count > counter.spec["encoder_max_tokens"],
        "embedding_eligible_by_length": token_count <= counter.spec["encoder_max_tokens"],
        "text": chunk_text, "start": start, "end": end,
        "pages": sorted({b["page_number"] for b in contributors}),
        "section_paths": [list(p) for p in sections], "mixed_sections": len(sections) > 1,
        "spans": spans, "review_status": "unreviewed_chunk_human_pending",
        "unresolved_objects": sorted({o for b in contributors for o in b["unresolved_objects"]}),
    }


def fixed_chunks(text, blocks, counter, *, budget=512, overlap=0):
    parameters = {"budget": budget, "overlap": overlap}
    return [make_chunk(text, blocks, counter, start, end, "fixed_tokens", parameters)
            for start, end, _ in counter.windows(text, budget, overlap)]


def structural_chunks(text, blocks, counter, *, budget=512, forced_breaks=()):
    """Agrupa párrafos por sección; conserva tablas multipágina aunque excedan ventana.

    forced_breaks son offsets opcionales propuestos por otro algoritmo, no títulos.
    """
    if budget < 16:
        raise ValueError("budget debe ser >=16.")
    atoms = []
    for block in blocks:
        if (atoms and block["kind"] == "table" and block["group_id"]
                and atoms[-1]["group_id"] == block["group_id"]):
            atoms[-1]["end"] = block["end"]
        else:
            atoms.append({"start": block["start"], "end": block["end"],
                          "kind": block["kind"], "group_id": block["group_id"],
                          "section_path": block["section_path"]})
    chunks, pending = [], None
    strategy = "semantic_optional" if forced_breaks else "structural"
    parameters = {"budget": budget, "tables": "atomic_linked_pages",
                  "forced_breaks": list(forced_breaks)}

    def emit(start, end):
        chunks.append(make_chunk(text, blocks, counter, start, end, strategy, parameters))

    for atom in atoms:
        start, end = atom["start"], atom["end"]
        if atom["kind"] == "table":
            if pending:
                emit(pending["start"], pending["end"])
                pending = None
            emit(start, end)
            continue
        pieces = [start, *sorted(p for p in forced_breaks if start < p < end), end]
        for left, right in zip(pieces, pieces[1:], strict=False):
            must_break = any(pending and pending["end"] <= p <= left for p in forced_breaks)
            compatible = pending and pending["section_path"] == atom["section_path"]
            fits = pending and counter.count(text[pending["start"]:right]) <= budget
            if pending and (not compatible or not fits or must_break or left in forced_breaks):
                emit(pending["start"], pending["end"])
                pending = None
            if counter.count(text[left:right]) > budget:
                for a, b, _ in counter.windows(text[left:right], budget):
                    emit(left + a, left + b)
            elif pending:
                pending["end"] = right
            else:
                pending = {"start": left, "end": right, "section_path": atom["section_path"]}
    if pending:
        emit(pending["start"], pending["end"])
    return chunks


def chunk_metrics(chunks: list[dict], text: str, blocks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("No se evalúa una colección vacía.")
    counts = [0] * (len(text) + 1)
    for chunk in chunks:
        counts[chunk["start"]] += 1
        counts[chunk["end"]] -= 1
    coverage, copies, unique, running = 0, 0, 0, 0
    for index, char in enumerate(text):
        running += counts[index]
        if not char.isspace():
            unique += 1
            coverage += running > 0
            copies += max(0, running)
    lengths = sorted(c["token_count_with_passage_prefix"] for c in chunks)
    table_groups = {}
    for block in blocks:
        if block["kind"] == "table":
            table_groups.setdefault(block["group_id"] or block["block_id"], []).append(block)
    intact = sum(any(c["start"] <= min(b["start"] for b in group)
                     and c["end"] >= max(b["end"] for b in group) for c in chunks)
                 for group in table_groups.values())
    return {
        "chunks": len(chunks), "tokens_min": lengths[0],
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_p95": lengths[min(len(lengths) - 1, math.ceil(len(lengths) * 0.95) - 1)],
        "tokens_max": lengths[-1],
        "coverage_of_selected_text": coverage / unique if unique else 1,
        "character_copy_factor": copies / coverage if coverage else 0,
        "chunks_over_encoder_window": sum(c["exceeds_encoder_window"] for c in chunks),
        "mixed_section_fraction": sum(c["mixed_sections"] for c in chunks) / len(chunks),
        "table_groups": len(table_groups), "intact_table_groups": intact,
        "intact_table_fraction": intact / len(table_groups) if table_groups else None,
        "paragraphs_split": sum(
            b["kind"] == "prose" and not any(
                c["start"] <= b["start"] and c["end"] >= b["end"] for c in chunks
            ) for b in blocks
        ),
    }


def token_overlap(a: str, b: str) -> float:
    """Cobertura multiconjunto de palabras: diagnóstico, no fidelidad semántica."""
    ca, cb = Counter(a.split()), Counter(b.split())
    return sum((ca & cb).values()) / sum(ca.values()) if ca else 1.0
