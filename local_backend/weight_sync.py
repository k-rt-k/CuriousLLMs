"""
LoRA hot-swap into a running vLLM OpenAI server.

vLLM 0.6.3+ exposes `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`
on the OpenAI server. After the trainer writes a fresh adapter directory, we
unload the previous adapter and load the new one, returning the name to use
in subsequent `model=` fields on `/v1/completions`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Union

import httpx

logger = logging.getLogger(__name__)


def _post(url: str, path: str, body: dict, timeout: float = 60.0) -> httpx.Response:
    return httpx.post(url.rstrip("/") + path, json=body, timeout=timeout)


def normalize_urls(urls: Union[str, Sequence[str], None]) -> list[str]:
    """A single URL, a comma-separated string, or a list -> list of stripped URLs."""
    if urls is None:
        return []
    if isinstance(urls, str):
        urls = urls.split(",")
    return [u.strip().rstrip("/") for u in urls if u and u.strip()]


def _fan_out(fn, urls: list[str]) -> None:
    """Run fn(url) for every url concurrently (callers are sync or asyncio.to_thread,
    so threads, not asyncio). Every URL is attempted; raise RuntimeError listing all failures."""
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [(u, pool.submit(fn, u)) for u in urls]
    errors = []
    for u, f in futures:
        exc = f.exception()
        if exc is not None:
            errors.append(f"{u}: {exc}")
    if errors:
        raise RuntimeError("LoRA sync failed on " + "; ".join(errors))


def _unload_one(vllm_url: str, name: str) -> None:
    """Best-effort unload on one server. 404 is fine (already gone)."""
    try:
        r = _post(vllm_url, "/v1/unload_lora_adapter", {"lora_name": name})
        if r.status_code >= 400 and r.status_code != 404:
            logger.warning("unload_lora_adapter(%s) on %s -> %s: %s", name, vllm_url, r.status_code, r.text[:300])
    except Exception as e:
        logger.warning("unload_lora_adapter(%s) on %s raised %s", name, vllm_url, e)


def _load_one(vllm_url: str, name: str, adapter_dir: str) -> None:
    """Load `adapter_dir` under `name` on one server. Idempotent on `name`: if vLLM
    reports the adapter is already loaded, re-load it in place so weights match the
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
    logger.info("vLLM %s loaded LoRA %s from %s", vllm_url, name, adapter_dir)


def _require(vllm_urls) -> list[str]:
    urls = normalize_urls(vllm_urls)
    if not urls:
        raise RuntimeError("No vllm_urls configured: cannot sync LoRA adapters.")
    return urls


def unload_lora(vllm_urls: Union[str, Sequence[str]], name: str) -> None:
    """Best-effort unload on every server (never raises)."""
    urls = normalize_urls(vllm_urls)
    if urls:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(lambda u: _unload_one(u, name), urls))


def load_lora(vllm_urls: Union[str, Sequence[str]], name: str, adapter_dir: str) -> None:
    """Load the adapter on every server concurrently; raise if any server fails."""
    _fan_out(lambda u: _load_one(u, name, adapter_dir), _require(vllm_urls))


def sync_lora_to_vllm(
    adapter_dir: str,
    name: str,
    vllm_urls: Union[str, Sequence[str]],
    prev_name: Optional[str] = None,
) -> None:
    """Unload the previous adapter, load the new one, on every server. Idempotent on `name`."""
    urls = _require(vllm_urls)
    if prev_name and prev_name != name:
        unload_lora(urls, prev_name)
    load_lora(urls, name, adapter_dir)
