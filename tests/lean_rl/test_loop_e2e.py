"""End-to-end lean RL loop on a tiny model against a fake vLLM (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import signal

import httpx
import pytest

from local_backend import weight_sync
from lean_rl.config import LeanRLConfig
from lean_rl.loop import run


class FakeVLLM:
    """Random completions (token ids + logprobs) and 200 for LoRA load/unload."""

    def __init__(self, on_completion=None):
        self.lora_loads: list[dict] = []
        self.completions = 0
        self.models: list[str] = []
        self._on_completion = on_completion
        self._rng = random.Random(0)

    def completion(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.completions += 1
        self.models.append(body["model"])
        if self._on_completion is not None:
            self._on_completion(self.completions)
        choices = []
        for i in range(body["n"]):
            toks = [self._rng.randrange(2, 64) for _ in range(body["max_tokens"])]
            lps = [-self._rng.uniform(0.1, 5.0) for _ in toks]
            choices.append({
                "index": i, "text": "", "finish_reason": "length",
                "logprobs": {"tokens": [f"token_id:{t}" for t in toks], "token_logprobs": lps},
            })
        return httpx.Response(200, json={"choices": choices})

    def lora(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/load_lora_adapter":
            self.lora_loads.append(json.loads(request.content))
        return httpx.Response(200, json={})

    def install(self, monkeypatch) -> httpx.MockTransport:
        lora_transport = httpx.MockTransport(self.lora)

        def post(url, json=None, timeout=None):
            with httpx.Client(transport=lora_transport) as c:
                return c.post(url, json=json, timeout=timeout)

        monkeypatch.setattr(weight_sync.httpx, "post", post)
        return httpx.MockTransport(self.completion)


def _cfg(model_dir: str, log_path: str, **kw) -> LeanRLConfig:
    base = dict(
        model_name=model_dir,
        renderer_name="role_colon",
        env="arithmetic",
        group_size=2,
        groups_per_batch=2,
        max_tokens=6,
        lora_rank=4,
        learning_rate=1e-3,
        kl_penalty_coef=0.01,
        entropy_bonus_coef=0.01,
        use_rnd_curiosity=False,
        use_reasoning_rewards=False,
        save_every=1,
        vllm_urls="http://fake:1",
        log_path=log_path,
    )
    base.update(kw)
    return LeanRLConfig(**base)


def _metrics(log_path: str) -> list[dict]:
    with open(os.path.join(log_path, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ckpts(log_path: str) -> list[str]:
    return sorted(os.listdir(os.path.join(log_path, "ckpt")))


def test_two_steps_arithmetic_with_fake_vllm(tiny_model_dir, tmp_path, monkeypatch):
    fake = FakeVLLM()
    transport = fake.install(monkeypatch)
    log = str(tmp_path / "run")
    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=2), transport=transport))

    rows = _metrics(log)
    assert len(rows) == 2
    for row in rows:
        for key in ("kl_policy_base", "entropy/bonus_mean", "loss"):
            assert key in row, key
        assert math.isfinite(row["loss"]) and row["grad_norm"] > 0
    assert [r["step"] for r in rows] == [0, 1]
    assert os.path.isdir(os.path.join(log, "ckpt", "step_000002"))
    assert len(fake.lora_loads) == 2
    # step 0 samples from the base model, step 1 from the adapter synced after step 0
    assert fake.models[0] == tiny_model_dir
    assert fake.models[-1] == fake.lora_loads[0]["lora_name"]


def test_resume_continues_step_count(tiny_model_dir, tmp_path, monkeypatch):
    fake = FakeVLLM()
    transport = fake.install(monkeypatch)
    log = str(tmp_path / "run")
    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=1), transport=transport))
    assert len(_metrics(log)) == 1
    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=2), transport=transport))
    rows = _metrics(log)
    assert [r["step"] for r in rows] == [0, 1]  # the second run did exactly one step
    with open(os.path.join(log, "ckpt", "step_000002", "loop_state.json")) as f:
        assert json.load(f)["step"] == 2


def test_preemption_mid_rollout_saves_last_completed_step(tiny_model_dir, tmp_path, monkeypatch):
    calls_per_step = 2 * 2  # groups_per_batch * group_size single-sample requests

    def preempt(n):
        if n == calls_per_step + 1:  # first rollout request of step 1
            os.kill(os.getpid(), signal.SIGUSR1)

    fake = FakeVLLM(on_completion=preempt)
    transport = fake.install(monkeypatch)
    log = str(tmp_path / "run")
    cfg = _cfg(tiny_model_dir, log, num_batches=3, save_every=100)
    asyncio.run(run(cfg, transport=transport))  # returns normally (exit 0)

    assert len(_metrics(log)) == 1  # step 1 was abandoned
    assert _ckpts(log) == ["step_000001"]
    assert signal.getsignal(signal.SIGUSR1) is not None  # handlers restored, process alive

    fake._on_completion = None
    asyncio.run(run(cfg, transport=transport))
    assert [r["step"] for r in _metrics(log)] == [0, 1, 2]
    assert "step_000003" in _ckpts(log)


LAUNCHER_ARGS = [
    "model_name=m", "env=arithmetic", "group_size=4", "groups_per_batch=8", "learning_rate=7e-5",
    "max_tokens=512", "lora_rank=32", "loss_fn=ppo", "fsdp=True", "micro_batch_tokens=8192",
    "gpt_oss_lm_head=True", "vllm_urls=http://a:8000,http://b:8000", "log_path=/tmp/x",
    "num_batches=3", "save_every=5", "keep_checkpoints=2", "eval_every=20",
    "attn_implementation=eager",
]


def test_cli_accepts_launcher_keys_and_rejects_unsupported():
    import chz

    cfg = chz.entrypoint(LeanRLConfig, argv=LAUNCHER_ARGS)
    assert cfg.fsdp and cfg.gpt_oss_lm_head and cfg.num_batches == 3
    assert cfg.vllm_urls == "http://a:8000,http://b:8000"
    default = chz.entrypoint(LeanRLConfig, argv=["log_path=/tmp/x"])
    assert default.max_tokens == 2048 and default.use_rnd_curiosity and default.use_reasoning_rewards
    for bad in ("dataset_schedule=m-m", "num_substeps=2", "remove_constant_reward_groups=True"):
        with pytest.raises((Exception, SystemExit)):
            chz.entrypoint(LeanRLConfig, argv=["log_path=/tmp/x", bad])
