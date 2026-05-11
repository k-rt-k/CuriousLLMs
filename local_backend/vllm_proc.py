"""
Helpers to launch the vLLM OpenAI-compatible server as a subprocess.

In normal operation the SLURM batch script launches vLLM and the trainer
connects to it over HTTP — this module is mostly used by tests and by
in-process callers that want a vLLM server lifecycle bound to the Python
process.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def start_vllm_server(
    model: str,
    port: int,
    *,
    enable_lora: bool = True,
    max_loras: int = 2,
    max_lora_rank: int = 32,
    max_cpu_loras: int = 8,
    gpu_memory_utilization: float = 0.45,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    enforce_eager: bool = True,
    log_path: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    """
    Spawn vLLM's OpenAI api_server as a child process.

    The default `gpu_memory_utilization=0.45` leaves ~half of the GPU for the
    co-resident trainer process. Big-model mode should pass 0.85 + use the
    sleep/wake_up endpoints to release GPU memory during the training step.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
    ]
    if enable_lora:
        cmd += [
            "--enable-lora",
            "--max-loras", str(max_loras),
            "--max-lora-rank", str(max_lora_rank),
            "--max-cpu-loras", str(max_cpu_loras),
        ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if extra_args:
        cmd += extra_args

    if log_path is not None:
        log_file = open(log_path, "ab", buffering=0)
        stdout = stderr = log_file
    else:
        stdout = stderr = None

    logger.info("Launching vLLM: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=os.environ.copy())
    return proc


def wait_until_ready(url: str, timeout: float = 600.0, interval: float = 3.0) -> None:
    """Block until vLLM's /health returns 200 or `timeout` elapses."""
    health = url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = httpx.get(health, timeout=5.0)
            if r.status_code == 200:
                logger.info("vLLM ready at %s", url)
                return
        except Exception as e:
            last_err = e
        time.sleep(interval)
    raise RuntimeError(f"vLLM at {url} not ready after {timeout}s (last err: {last_err})")


def stop(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    """Best-effort terminate of the vLLM server."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)
