from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from radiology_rag.ingestion.chunking import (
    TokenCounter,
    chunk_metrics,
    fixed_chunks,
    flatten_blocks,
    structural_chunks,
)
from radiology_rag.ingestion.parsing import join_dropcap, paragraph_lines


def test_dropcap_rejoins_word_and_keeps_original_geometry():
    lines = [
        {"text": "M", "x0": 10, "x1": 20, "top": 10, "bottom": 40, "chars": ["M"]},
        {"text": "agnetic resonance", "x0": 21, "x1": 150,
         "top": 20, "bottom": 30, "chars": ["agnetic"]},
    ]
    result = join_dropcap(lines)
    assert result[0]["text"] == "Magnetic resonance"
    assert (result[0]["x0"], result[0]["top"], result[0]["bottom"]) == (10, 10, 40)
    assert result[0]["chars"] == ["M", "agnetic"]
    assert lines[0]["text"] == "M"


def test_dropcap_does_not_join_unrelated_lines():
    lines = [
        {"text": "A", "x0": 10, "x1": 20, "top": 10, "bottom": 20},
        {"text": "another paragraph", "x0": 10, "x1": 100, "top": 40, "bottom": 50},
    ]
    assert join_dropcap(lines) == lines


@pytest.fixture
def counter():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return TokenCounter(tokenizer, {
        "passage_prefix": "passage: ", "encoder_max_tokens": 32,
        "repository": "unit-test-only", "revision": "test", "sha256": "test",
    })


def block(text, identifier, *, kind="prose", page=1, section="A", group=None):
    return {
        "text": text, "block_id": identifier, "kind": kind, "group_id": group,
        "page_number": page, "page_label": str(page + 700), "region_id": identifier,
        "bbox": [0, 0, 100, 100], "section_path": [section], "unresolved_objects": [],
        "document_id": "test-doc", "document_version": "test-version",
        "document_role": "test-only", "source_url": "https://example.test",
        "source_sha256": "test-hash", "parser_fingerprint": "test-parser",
    }


@pytest.mark.parametrize("overlap", [0, 0.15])
def test_fixed_token_budget_coverage_and_exact_source_spans(counter, overlap):
    blocks = [block(" ".join(f"word{i}" for i in range(120)), "one"),
              block("áéí español 漢字 🙂 with original punctuation.", "two", page=2)]
    text, located = flatten_blocks(blocks)
    chunks = fixed_chunks(text, located, counter, budget=32, overlap=overlap)
    assert all(counter.count(c["text"]) <= 32 for c in chunks)
    assert all(c["text"] == text[c["start"]:c["end"]] for c in chunks)
    assert all(c["spans"] and c["pages"] for c in chunks)
    metrics = chunk_metrics(chunks, text, located)
    assert metrics["coverage_of_selected_text"] == 1
    if overlap:
        assert metrics["character_copy_factor"] > 1
    else:
        assert metrics["character_copy_factor"] == 1


def test_linked_table_stays_whole_and_oversize_is_flagged(counter):
    blocks = [block("Table title " + "data " * 30, "t1", kind="table", group="T", page=8),
              block("Continuation and note " + "data " * 30, "t2", kind="table", group="T", page=9)]
    text, located = flatten_blocks(blocks)
    chunks = structural_chunks(text, located, counter, budget=32)
    assert len(chunks) == 1 and chunks[0]["pages"] == [8, 9]
    assert chunks[0]["exceeds_encoder_window"]
    assert not chunks[0]["embedding_eligible_by_length"]
    assert chunks[0]["text"] == text
    assert chunk_metrics(chunks, text, located)["intact_table_fraction"] == 1


def test_structural_does_not_mix_sections_and_preserves_all_text(counter):
    text, blocks = flatten_blocks([block("alpha beta gamma", "a", section="A"),
                                 block("delta epsilon zeta", "b", section="B")])
    chunks = structural_chunks(text, blocks, counter, budget=32)
    assert len(chunks) == 2
    assert not any(c["mixed_sections"] for c in chunks)
    assert chunk_metrics(chunks, text, blocks)["coverage_of_selected_text"] == 1


def test_ids_are_deterministic_and_include_parameters_and_version(counter):
    text, blocks = flatten_blocks([block("data " * 50, "a")])
    first = fixed_chunks(text, blocks, counter, budget=32)
    assert first == fixed_chunks(text, blocks, counter, budget=32)
    assert first[0]["chunk_id"] != fixed_chunks(text, blocks, counter, budget=24)[0]["chunk_id"]
    changed = [{**b, "document_version": "another-version"} for b in blocks]
    assert first[0]["chunk_id"] != fixed_chunks(text, changed, counter, budget=32)[0]["chunk_id"]


def test_references_and_metadata_are_preserved_as_blocks_but_not_retrieval_text():
    text, located = flatten_blocks([block("body", "a"), block("citation", "r", kind="reference"),
                                  block("license", "m", kind="metadata")])
    assert text == "body" and len(located) == 1


def test_heading_attaches_to_following_paragraph():
    def line(text, top, bold=False):
        return {"text": text, "top": top, "bottom": top + 10,
                "chars": [{"fontname": "Font-Bold" if bold else "Font-Regular"}] * len(text)}
    groups = paragraph_lines([line("Heading", 0, True), line("Text", 25), line("Next", 50)])
    assert [[item["text"] for item in group] for group in groups] == [["Heading", "Text"], ["Next"]]


def test_real_pinned_tokenizer_unicode_offsets_if_prepared():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/raw/tokenizers/e5-base-d128750/tokenizer.json").exists():
        pytest.skip("Tokenizer real no preparado: los tests unitarios siguen siendo offline.")
    counter = TokenCounter.from_project(root)
    text = "Árbol, ADC ≥ b800; 漢字 🙂. " * 60
    windows = list(counter.windows(text, 64, 0.15))
    assert all(counter.count(text[a:b]) <= 64 for a, b, _ in windows)
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(next_a <= b for (_, b, _), (next_a, _, _) in zip(windows, windows[1:], strict=False))


def test_invalid_token_window_configuration_is_rejected(counter):
    with pytest.raises(ValueError):
        list(counter.windows("text", 8))
    with pytest.raises(ValueError):
        list(counter.windows("text", 32, 1))


def test_structural_forced_breaks_preserve_text_and_change_boundaries(counter):
    text, blocks = flatten_blocks([block("alpha beta gamma", "a"), block("delta epsilon", "b")])
    assert len(structural_chunks(text, blocks, counter, budget=32)) == 1
    chunks = structural_chunks(text, blocks, counter, budget=32, forced_breaks=[blocks[1]["start"]])
    assert len(chunks) == 2
    assert chunk_metrics(chunks, text, blocks)["coverage_of_selected_text"] == 1
