import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "lean_rl" / "vllm_plugin"


def test_register_sets_embedding_modules():
    try:
        from vllm.model_executor.models.gpt_oss import GptOssForCausalLM
    except Exception as e:  # vLLM missing or not importable here
        pytest.skip(f"vLLM import failed: {e}")
    sys.path.insert(0, str(PLUGIN_DIR))
    try:
        from lean_rl_vllm_plugin import register
    finally:
        sys.path.remove(str(PLUGIN_DIR))
    register()
    assert GptOssForCausalLM.embedding_modules["lm_head"] == "output_embeddings"
    register()  # idempotent
    assert GptOssForCausalLM.embedding_modules == {"lm_head": "output_embeddings"}
