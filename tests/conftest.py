"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    """A tiny random Llama + word-level tokenizer saved to disk. No network."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_llama")
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")),
        eos_token="t1",
        pad_token="t1",
    )
    tok.save_pretrained(d)
    return str(d)
