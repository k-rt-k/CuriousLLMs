"""End-to-end lean RL loop on a tiny model against a fake vLLM (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import signal
import threading

import httpx
import pytest
import torch

from local_backend import weight_sync
from lean_rl import loop as loop_mod
from lean_rl.config import LeanRLConfig
from lean_rl.loop import run


class FakeVLLM:
    """Random completions (token ids + logprobs) and 200 for LoRA load/unload. Once ``dead``
    (vLLM SIGTERMed by the preemption) every request raises ``httpx.ConnectError``."""

    def __init__(self, on_completion=None, on_lora_load=None):
        self.lora_loads: list[dict] = []
        self.completions = 0
        self.models: list[str] = []
        self.dead = False
        self._on_completion = on_completion
        self._on_lora_load = on_lora_load
        self._rng = random.Random(0)

    def _check_alive(self) -> None:
        if self.dead:
            raise httpx.ConnectError("fake vLLM is gone")

    def completion(self, request: httpx.Request) -> httpx.Response:
        self._check_alive()
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
            if self._on_lora_load is not None:
                self._on_lora_load(len(self.lora_loads))
        self._check_alive()
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


class _PreemptedEval:
    """Evaluator whose second call is preempted: SIGUSR1 and vLLM dies; the call then either
    hits the dead fake vLLM (raises) or hangs (the loop must abandon it by polling)."""

    def __init__(self, fake: FakeVLLM, mode: str):
        self.fake, self.mode, self.calls = fake, mode, 0

    async def __call__(self, sampling_client) -> dict:
        import tinker

        self.calls += 1
        if self.calls == 2:
            self.fake.dead = True
            os.kill(os.getpid(), signal.SIGUSR1)
            if self.mode == "hang":
                await asyncio.sleep(3600)
        await sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints([2, 3]), num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=2),
        )
        return {"acc": 1.0}


@pytest.mark.parametrize("mode", ["raise", "hang"])
def test_preemption_during_eval_saves_last_completed_step(tiny_model_dir, tmp_path, monkeypatch, mode):
    fake = FakeVLLM()
    transport = fake.install(monkeypatch)
    ev = _PreemptedEval(fake, mode)
    monkeypatch.setattr(loop_mod, "_make_evaluators", lambda *a: [ev])
    log = str(tmp_path / "run")
    cfg = _cfg(tiny_model_dir, log, num_batches=3, save_every=100, eval_every=1)
    asyncio.run(run(cfg, transport=transport))  # returns normally (exit 0)

    rows = _metrics(log)
    assert [r["step"] for r in rows] == [0] and rows[0]["test/acc"] == 1.0
    assert ev.calls == 2
    assert _ckpts(log) == ["step_000001"]


def test_vllm_sync_failure_during_preemption_saves_completed_step(tiny_model_dir, tmp_path, monkeypatch):
    """vLLM dies during the adapter sync after step 1 and our SIGTERM lands only afterwards:
    the failure is treated as preemption and the completed step is saved."""

    def on_load(n):
        if n == 2:
            fake.dead = True
            threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGUSR1)).start()

    fake = FakeVLLM(on_lora_load=on_load)
    transport = fake.install(monkeypatch)
    log = str(tmp_path / "run")
    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=3, save_every=100), transport=transport))
    assert [r["step"] for r in _metrics(log)] == [0, 1]
    assert _ckpts(log) == ["step_000002"]


def test_vllm_sync_failure_without_preemption_raises(tiny_model_dir, tmp_path, monkeypatch):
    fake = FakeVLLM(on_lora_load=lambda n: setattr(fake, "dead", n == 2))
    transport = fake.install(monkeypatch)
    monkeypatch.setattr(loop_mod, "_PREEMPT_GRACE_S", 0.2)
    log = str(tmp_path / "run")
    with pytest.raises(RuntimeError, match="LoRA sync failed"):
        asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=3, save_every=100), transport=transport))


def test_resume_skips_unloadable_checkpoint_and_stale_tmp(tiny_model_dir, tmp_path, monkeypatch):
    fake = FakeVLLM()
    transport = fake.install(monkeypatch)
    log = str(tmp_path / "run")
    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=2, keep_checkpoints=2), transport=transport))
    assert _ckpts(log) == ["step_000001", "step_000002"]
    with open(os.path.join(log, "ckpt", "step_000002", "optimizer.pt"), "wb") as f:
        f.write(b"truncated")  # newest checkpoint no longer loads
    stale = os.path.join(log, "ckpt", "step_000003.tmp")
    os.makedirs(stale)
    open(os.path.join(stale, "optimizer.pt"), "w").close()

    asyncio.run(run(_cfg(tiny_model_dir, log, num_batches=3, keep_checkpoints=2), transport=transport))
    assert [r["step"] for r in _metrics(log)] == [0, 1, 1, 2]  # resumed after step 1
    assert not os.path.exists(stale)
    assert _ckpts(log) == ["step_000002", "step_000003"]
    torch.load(os.path.join(log, "ckpt", "step_000002", "optimizer.pt"), weights_only=False)


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


def test_missing_rnd_embeddings_fail_at_startup(tiny_model_dir, tmp_path, monkeypatch):
    """RND is built only after the model and vLLM are up (20+ min on Babel); a missing
    embeddings file must stop the run before any of that."""
    from lean_rl import loop as loop_mod

    monkeypatch.chdir(tmp_path)  # no embeddings/ here
    built = []
    monkeypatch.setattr(loop_mod._Loop, "_build_trainer", lambda self: built.append(1))
    cfg = _cfg(tiny_model_dir, str(tmp_path / "log"), use_rnd_curiosity=True)
    with pytest.raises(FileNotFoundError, match="use_rnd_curiosity=False"):
        asyncio.run(run(cfg, transport=None))
    assert not built
