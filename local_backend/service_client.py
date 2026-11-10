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
- `base_url` is repurposed as the vLLM HTTP endpoint(s) (comma-separated for several).
- `model_path` is a local PEFT adapter directory.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Union

from local_backend.distributed import get_context
from local_backend.sampling_client import LocalSamplingClient
from local_backend.training_client import LocalTrainingClient
from local_backend.weight_sync import load_lora, normalize_urls

logger = logging.getLogger(__name__)


class LocalServiceClient:
    """
    Owns the vLLM URL + tokenizer cache. Hands out training and sampling clients.

    Args:
        vllm_urls: URL(s) of running vLLM OpenAI-compatible servers (list, or
                  comma-separated string). If None, read $VLLM_URLS (comma-separated),
                  then $VLLM_URL (set by sbatch_train.sh).
        base_model_name: the HF model id loaded into vLLM. Sampling clients
                  bound to "no adapter" will route to this.
    """

    def __init__(
        self,
        *,
        vllm_urls: Union[str, Sequence[str], None] = None,
        base_model_name: Optional[str] = None,
        log_path: Optional[str] = None,
        **_unused,  # accept and discard `base_url` etc. for back-compat
    ):
        if vllm_urls is None:
            vllm_urls = (
                os.environ.get("VLLM_URLS")
                or os.environ.get("VLLM_URL")
                or _unused.get("base_url")
            )
        self.vllm_urls = normalize_urls(vllm_urls)
        if not self.vllm_urls:
            raise RuntimeError(
                "LocalServiceClient requires vllm_urls. Pass vllm_urls=... or set $VLLM_URLS / $VLLM_URL."
            )
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
        Construct a LocalTrainingClient that fine-tunes `base_model` with LoRA(r=rank).
        Memory share with vLLM is configured via
        torch.cuda.set_per_process_memory_fraction inside LocalTrainingClient.
        Unless `dist=` is passed, the client joins the process-global
        data-parallel context from `local_backend.distributed.init_distributed()`
        (None when single-process).
        """
        self.base_model_name = self.base_model_name or base_model
        if "dist" not in kwargs:
            kwargs["dist"] = get_context()
        client = LocalTrainingClient(
            base_model=base_model,
            lora_rank=rank,
            vllm_urls=self.vllm_urls,
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
        adapter_name: Optional[str] = None,
        **_unused,
    ) -> LocalSamplingClient:
        """
        If `model_path` is set, it's a local PEFT adapter directory: load it into
        vLLM under a name derived from the path and bind the returned client to
        that adapter.
        `adapter_name` overrides the default path-leaf adapter name (e.g. the
        fixed "kl_ref" used for the KL reference adapter).
        If only `base_model` is set, bind to the base model (no adapter).
        """
        bm = base_model or self.base_model_name
        if bm is None:
            raise RuntimeError("create_sampling_client needs base_model or service must know it.")
        self.base_model_name = bm

        if model_path is not None:
            adapter_name = adapter_name or _adapter_name_from_path(model_path)
            try:
                load_lora(self.vllm_urls, adapter_name, model_path)
            except RuntimeError as e:
                # Already loaded? Best-effort idempotency.
                logger.warning("load_lora during create_sampling_client raised: %s", e)

        return LocalSamplingClient(
            vllm_urls=self.vllm_urls,
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
