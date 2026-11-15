from xml.etree import ElementTree as ET

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from radiology_rag.ingestion.chunking import TokenCounter
from radiology_rag.ingestion.jats import build_passages, expand_rows, row_attributes, table_records


def test_merged_cells_expand_without_losing_row_or_column_associations():
    rows = ET.fromstring('''<tbody>
      <tr><td rowspan="2">shared</td><td>A</td><td>B</td></tr>
      <tr><td colspan="2">combined</td></tr></tbody>''')
    assert expand_rows(rows) == [["shared", "A", "B"], ["shared", "combined", "combined"]]


def test_jats_indexes_body_and_float_tables_with_headers_notes_and_provenance():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    counter = TokenCounter(tokenizer, {"passage_prefix": "passage: ",
                                     "encoder_max_tokens": 512, "sha256": "test"})
    xml = b'''<article><body><sec id="s1"><title>Unit section</title>
      <p id="p1">Unit prose.</p></sec></body>
      <back><ref>Excluded reference text</ref></back><floats-group>
      <table-wrap id="t1"><label>Table 1</label><caption><p>Unit table</p></caption>
      <table><thead><tr><th>Condition</th><th>Result</th></tr></thead>
      <tbody><tr><td>Alpha</td><td>Beta</td></tr></tbody></table>
      <table-wrap-foot><p>Except when unit condition.</p></table-wrap-foot></table-wrap>
      <fig id="f1"><caption><p>Uninterpreted image</p></caption></fig>
      </floats-group></article>'''
    doc = {"id": "unit", "version": "test", "system": "BI-RADS", "role": "secondary",
           "source_url": "https://example.test", "expected_sha256": "pdf-test",
           "repository_xml_sha256": "xml-test"}
    passages, objects = build_passages(xml, doc, counter)
    assert len(passages) == 3 and len(objects) == 2
    row = passages[1]
    assert '1. Condition' in row["text"] and 'Alpha' in row["text"]
    assert '2. Result' in row["text"] and 'Beta' in row["text"]
    assert 'Except when unit condition' in row["text"]
    assert row["xml_element_id"] == "t1" and row["table_row"] == 1
    assert row["pages"] == [] and row["xml_sha256"] == "xml-test"
    assert all(p["token_count_with_passage_prefix"] <= 512 for p in passages)
    assert all('Excluded reference' not in p["text"] and 'Uninterpreted image' not in p["text"]
               for p in passages)
    assert build_passages(xml, doc, counter) == (passages, objects)


def test_internal_publisher_headers_replace_previous_columns():
    wrapper = ET.fromstring('''<table-wrap><table><thead>
      <tr><th>Old A</th><th>Old B</th></tr></thead><tbody>
      <tr><td>Value A</td><td>Value B</td></tr>
      <tr><td colspan="2"><bold>Second section</bold></td></tr>
      <tr><td><bold>New A</bold></td><td><bold>New B</bold></td></tr>
      <tr><td>New value A</td><td>New value B</td></tr>
      </tbody></table></table-wrap>''')
    rows = table_records(wrapper)
    assert len(rows) == 2
    assert rows[1]["headers"] == ["New A", "New B"]
    assert rows[1]["cells"] == ["New value A", "New value B"]
    assert rows[1]["row_number"] == 4 and rows[1]["section"] == "Second section"


def test_categories_come_only_from_explicit_category_cells():
    row = {"headers": ["Category", "Year"], "cells": ["3", "2025"], "section": "unit"}
    assert row_attributes(row)["assessment_categories"] == ["3"]
    row["headers"] = ["Finding Category", "Year"]
    assert row_attributes(row)["assessment_categories"] == []
