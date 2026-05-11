"""
LoRA hot-swap into a running vLLM OpenAI server.

vLLM 0.6.3+ exposes `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`
on the OpenAI server. After the trainer writes a fresh adapter directory, we
unload the previous adapter and load the new one, returning the name to use
in subsequent `model=` fields on `/v1/completions`.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _post(url: str, path: str, body: dict, timeout: float = 60.0) -> httpx.Response:
    return httpx.post(url.rstrip("/") + path, json=body, timeout=timeout)


def unload_lora(vllm_url: str, name: str) -> None:
    """Best-effort unload. 404 is fine (already gone)."""
    try:
        r = _post(vllm_url, "/v1/unload_lora_adapter", {"lora_name": name})
        if r.status_code >= 400 and r.status_code != 404:
            logger.warning("unload_lora_adapter(%s) -> %s: %s", name, r.status_code, r.text[:300])
    except Exception as e:
        logger.warning("unload_lora_adapter(%s) raised %s", name, e)


def load_lora(vllm_url: str, name: str, adapter_dir: str) -> None:
    """Load `adapter_dir` under `name`. Idempotent on `name`: if vLLM reports
    the adapter is already loaded, re-load it in place so weights match the
    fresh files on disk.
    """
    body = {"lora_name": name, "lora_path": adapter_dir}
    r = _post(vllm_url, "/v1/load_lora_adapter", body, timeout=180.0)
    if r.status_code == 400 and "already been loaded" in r.text:
        # vLLM 0.15+: opt into in-place reload so we pick up new weights.
        body["load_inplace"] = True
        r = _post(vllm_url, "/v1/load_lora_adapter", body, timeout=180.0)
    if r.status_code >= 400:
        raise RuntimeError(
            f"load_lora_adapter(name={name}, path={adapter_dir}) -> "
            f"{r.status_code}: {r.text[:500]}"
        )
    logger.info("vLLM loaded LoRA %s from %s", name, adapter_dir)


def sync_lora_to_vllm(
    adapter_dir: str,
    name: str,
    vllm_url: str,
    prev_name: Optional[str] = None,
) -> None:
    """Unload the previous adapter, load the new one. Idempotent on `name`."""
    if prev_name and prev_name != name:
        unload_lora(vllm_url, prev_name)
    load_lora(vllm_url, name, adapter_dir)
