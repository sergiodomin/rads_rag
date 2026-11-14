import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from radiology_rag.generation.context import context_budget


def test_context_reserves_output_and_rejects_oversize_without_truncation():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    messages = [{"role": "system", "content": "Use only sources"},
                {"role": "user", "content": "short question"}]
    report = context_budget(messages, tokenizer, num_ctx=1024, num_predict=128)
    assert 128 < report["context_tokens_budgeted"] < 896
    messages[-1]["content"] = "token " * 1000
    with pytest.raises(ValueError, match="no se ha truncado"):
        context_budget(messages, tokenizer, num_ctx=1024, num_predict=128)
    assert messages[-1]["content"] == "token " * 1000
