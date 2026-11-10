"""vLLM general plugin: allow LoRA on gpt-oss ``lm_head``.

vLLM's ``GptOssForCausalLM`` declares no ``embedding_modules``, so the LoRA manager
does not list ``lm_head`` as a supported module and rejects adapters that include it
(Qwen3 declares ``{"embed_tokens": ..., "lm_head": "output_embeddings"}``).

vLLM loads general plugins in every process (API server, EngineCore, workers), so
``register`` must be idempotent and import-light. Enable with
``VLLM_PLUGINS=lean_rl_gpt_oss_lm_head`` after ``pip install -e lean_rl/vllm_plugin``.
"""

from __future__ import annotations


def register() -> None:
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    modules = dict(getattr(GptOssForCausalLM, "embedding_modules", None) or {})
    modules["lm_head"] = "output_embeddings"
    GptOssForCausalLM.embedding_modules = modules
