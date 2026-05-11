"""
LocalServiceClient — duck-typed replacement for `tinker.ServiceClient`.

Tinker call shape (curiosity_train.py:2106-2117, math_evaluation.py):

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )
    # Or for eval / KL-reference:
    sampling_client = service_client.create_sampling_client(
        model_path=path,            # tinker remote path
        base_model=model_name,
    )

In the local backend:
- `base_url` is repurposed as `vllm_url` (the running vLLM HTTP endpoint).
- `model_path` is a local PEFT adapter directory.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from local_backend.sampling_client import LocalSamplingClient
from local_backend.training_client import LocalTrainingClient
from local_backend.weight_sync import load_lora

logger = logging.getLogger(__name__)


class LocalServiceClient:
    """
    Owns the vLLM URL + tokenizer cache. Hands out training and sampling clients.

    Args:
        vllm_url: URL of a running vLLM OpenAI-compatible server.
                  If None, read from the VLLM_URL env var (set by sbatch_train.sh).
        base_model_name: the HF model id loaded into vLLM. Sampling clients
                  bound to "no adapter" will route to this.
    """

    def __init__(
        self,
        *,
        vllm_url: Optional[str] = None,
        base_model_name: Optional[str] = None,
        log_path: Optional[str] = None,
        **_unused,  # accept and discard `base_url` etc. for back-compat
    ):
        if vllm_url is None:
            vllm_url = os.environ.get("VLLM_URL") or _unused.get("base_url")
        if vllm_url is None:
            raise RuntimeError(
                "LocalServiceClient requires vllm_url. Pass vllm_url=... or set $VLLM_URL."
            )
        self.vllm_url = vllm_url.rstrip("/")
        self.base_model_name = base_model_name  # may be filled in later by callers
        self.log_path = log_path or os.environ.get("LOG_DIR")
        self._tokenizer_cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # TrainingClient creation
    # ------------------------------------------------------------------
    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        **kwargs,
    ) -> LocalTrainingClient:
        """
        Construct a LocalTrainingClient that fine-tunes `base_model` with LoRA(r=rank)
        on the single visible GPU. Memory share with vLLM is configured via
        torch.cuda.set_per_process_memory_fraction inside LocalTrainingClient.
        """
        self.base_model_name = self.base_model_name or base_model
        client = LocalTrainingClient(
            base_model=base_model,
            lora_rank=rank,
            vllm_url=self.vllm_url,
            log_path=self.log_path,
            **kwargs,
        )
        await client._init_async()
        return client

    def create_lora_training_client(self, *args, **kwargs):
        import asyncio
        return asyncio.run(self.create_lora_training_client_async(*args, **kwargs))

    # ------------------------------------------------------------------
    # SamplingClient creation
    # ------------------------------------------------------------------
    def create_sampling_client(
        self,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
        **_unused,
    ) -> LocalSamplingClient:
        """
        If `model_path` is set, it's a local PEFT adapter directory: load it into
        vLLM under a name derived from the path and bind the returned client to
        that adapter.
        If only `base_model` is set, bind to the base model (no adapter).
        """
        bm = base_model or self.base_model_name
        if bm is None:
            raise RuntimeError("create_sampling_client needs base_model or service must know it.")
        self.base_model_name = bm

        adapter_name: Optional[str] = None
        if model_path is not None:
            adapter_name = _adapter_name_from_path(model_path)
            try:
                load_lora(self.vllm_url, adapter_name, model_path)
            except RuntimeError as e:
                # Already loaded? Best-effort idempotency.
                logger.warning("load_lora during create_sampling_client raised: %s", e)

        return LocalSamplingClient(
            vllm_url=self.vllm_url,
            base_model_name=bm,
            adapter_name=adapter_name,
        )

    async def create_sampling_client_async(self, *args, **kwargs):
        return self.create_sampling_client(*args, **kwargs)


def _adapter_name_from_path(model_path: str) -> str:
    """
    Derive a stable vLLM adapter name from a local directory path. We use the
    leaf directory name (e.g., `step_000010`); if non-unique across runs this
    is fine because we unload before reloading.
    """
    leaf = os.path.basename(model_path.rstrip("/"))
    return leaf or "lora_adapter"
