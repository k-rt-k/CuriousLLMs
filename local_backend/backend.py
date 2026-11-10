"""Backend selection: Tinker (network) or local (vLLM + HF/PEFT)."""

from __future__ import annotations

from typing import Literal


def make_service_client(
    backend: Literal["tinker", "local"],
    *,
    base_url: str | None,
    log_path: str | None,
    model_name: str,
):
    """Return a `tinker.ServiceClient` or a duck-typed `LocalServiceClient`."""
    if backend == "tinker":
        import tinker  # lazy: only needed for the Tinker backend

        return tinker.ServiceClient(base_url=base_url)
    if backend == "local":
        import os

        from local_backend import LocalServiceClient

        return LocalServiceClient(
            vllm_url=base_url or os.environ.get("VLLM_URL"),
            log_path=log_path,
            base_model_name=model_name,
        )
    raise ValueError(f"Unknown backend {backend!r}; expected 'tinker' or 'local'")
