import os
import random
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

tinker = pytest.importorskip("tinker")

from lean_rl.checkpoint import (  # noqa: E402
    LoopState, PreemptionFlag, load_latest, remove_stale_tmp, restore_checkpoint, save_checkpoint,
)
from tests.lean_rl.test_trainer import _make_data, _trainer  # noqa: E402

ADAM = tinker.AdamParams(learning_rate=1e-2, beta1=0.9, beta2=0.95, eps=1e-8)


def _steps(tr, n):
    for _ in range(n):
        tr.train_step(_make_data(seed=tr.step), "importance_sampling", ADAM)


def _assert_lora_close(a, b, atol=1e-6):
    sa, sb = a.state_dict_for_ckpt()["lora"], b.state_dict_for_ckpt()["lora"]
    assert sa.keys() == sb.keys()
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k], atol=atol, rtol=0)


def test_roundtrip_restores_identical_training(tiny_model_dir, tmp_path):
    ref = _trainer(tiny_model_dir)
    _steps(ref, 3)

    a = _trainer(tiny_model_dir)
    _steps(a, 2)
    path = save_checkpoint(str(tmp_path), 2, a, LoopState(2, 7, "run"), {"x": torch.ones(2)})
    assert sorted(os.listdir(path)) == [
        "adapter", "loop_state.json", "optimizer.pt", "rnd.pt", "rng.pt"]

    found = load_latest(str(tmp_path))
    assert found is not None and found[0] == path
    assert found[1] == LoopState(2, 7, "run")
    b = _trainer(tiny_model_dir)
    rnd = restore_checkpoint(found[0], b)
    assert torch.equal(rnd["x"], torch.ones(2))
    assert b.step == 2
    _steps(b, 1)
    _assert_lora_close(b, ref)


def test_rng_roundtrip(tiny_model_dir, tmp_path):
    tr = _trainer(tiny_model_dir)
    save_checkpoint(str(tmp_path), 0, tr, LoopState(0, 0, None), None)
    expect = (torch.rand(3), np.random.rand(3), random.random())  # draws after the save
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)  # scramble
    restore_checkpoint(load_latest(str(tmp_path))[0], tr)
    got = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(got[0], expect[0]) and (got[1] == expect[1]).all() and got[2] == expect[2]


def test_retention_keeps_last_k(tiny_model_dir, tmp_path):
    tr = _trainer(tiny_model_dir)
    for s in range(1, 5):
        save_checkpoint(str(tmp_path), s, tr, LoopState(s, s, None), None, keep=2)
    assert sorted(os.listdir(tmp_path / "ckpt")) == ["step_000003", "step_000004"]
    assert load_latest(str(tmp_path))[1].step == 4


def test_non_main_rank_does_not_write(tiny_model_dir, tmp_path):
    tr = _trainer(tiny_model_dir)
    save_checkpoint(str(tmp_path), 1, tr, LoopState(1, 1, None), None, is_main=False)
    assert load_latest(str(tmp_path)) is None


def test_crash_during_save_keeps_previous(tiny_model_dir, tmp_path, monkeypatch):
    tr = _trainer(tiny_model_dir)
    save_checkpoint(str(tmp_path), 1, tr, LoopState(1, 1, None), None)

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_checkpoint(str(tmp_path), 2, tr, LoopState(2, 2, None), None)
    monkeypatch.undo()
    assert "step_000002.tmp" in os.listdir(tmp_path / "ckpt")
    found = load_latest(str(tmp_path))
    assert found is not None and found[1].step == 1


def test_load_latest_falls_back_to_older_loadable(tiny_model_dir, tmp_path):
    tr = _trainer(tiny_model_dir)
    _steps(tr, 1)
    save_checkpoint(str(tmp_path), 1, tr, LoopState(1, 1, None), None)
    _steps(tr, 1)
    save_checkpoint(str(tmp_path), 2, tr, LoopState(2, 2, None), None)
    with open(tmp_path / "ckpt" / "step_000002" / "optimizer.pt", "wb") as f:
        f.write(b"corrupt")
    found = load_latest(str(tmp_path))
    assert found is not None and found[1].step == 1
    b = _trainer(tiny_model_dir)
    restore_checkpoint(found[0], b)
    assert b.step == 1
    with open(tmp_path / "ckpt" / "step_000001" / "rng.pt", "wb") as f:
        f.write(b"corrupt")
    with pytest.raises(RuntimeError, match="No loadable checkpoint"):
        load_latest(str(tmp_path))


def test_remove_stale_tmp(tiny_model_dir, tmp_path):
    assert remove_stale_tmp(str(tmp_path)) == []  # no ckpt dir yet
    tr = _trainer(tiny_model_dir)
    save_checkpoint(str(tmp_path), 1, tr, LoopState(1, 1, None), None)
    for name in ("step_000002.tmp", "step_000003.tmp"):
        (tmp_path / "ckpt" / name).mkdir()
        (tmp_path / "ckpt" / name / "optimizer.pt").write_bytes(b"partial")
    removed = remove_stale_tmp(str(tmp_path))
    assert sorted(os.path.basename(p) for p in removed) == ["step_000002.tmp", "step_000003.tmp"]
    assert os.listdir(tmp_path / "ckpt") == ["step_000001"]


def test_preemption_flag_sets_on_signal():
    old = {s: signal.getsignal(s) for s in PreemptionFlag.SIGNALS}
    try:
        flag = PreemptionFlag().install()
        assert flag.requested is False
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        assert flag.requested is True
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_preemption_flag_sigint_only_when_enabled(monkeypatch):
    sigs = (*PreemptionFlag.SIGNALS, signal.SIGINT)
    old = {s: signal.getsignal(s) for s in sigs}
    try:
        monkeypatch.delenv("LEAN_PREEMPT_ON_SIGINT", raising=False)
        PreemptionFlag().install()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)

        monkeypatch.setenv("LEAN_PREEMPT_ON_SIGINT", "1")
        flag = PreemptionFlag().install()
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        assert flag.requested is True
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _run(args):
    return subprocess.Popen([sys.executable, "-m", "tests.lean_rl.sigterm_loop", *args])


def test_sigterm_mid_step_resumes_consistently(tiny_model_dir, tmp_path):
    base = ["--model", tiny_model_dir]
    ref_out = str(tmp_path / "ref.pt")
    assert _run(base + ["--log", str(tmp_path / "ref"), "--out", ref_out]).wait(600) == 0

    log, out, marker = str(tmp_path / "run"), str(tmp_path / "run.pt"), str(tmp_path / "marker")
    p = _run(base + ["--log", log, "--out", out, "--marker", marker, "--wait-step", "1"])
    t0 = time.time()
    while not os.path.exists(marker):
        assert p.poll() is None and time.time() - t0 < 300, "loop died before marker"
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGTERM)
    assert p.wait(120) == 0
    assert not os.path.exists(out)  # exited at the boundary, not finished
    found = load_latest(log)
    assert found is not None and found[1].step == 2

    assert _run(base + ["--log", log, "--out", out]).wait(600) == 0
    a, b = torch.load(out), torch.load(ref_out)
    assert a["step"] == b["step"] == 5 and a["pos"] == b["pos"] == 5
    for k in b["lora"]:
        torch.testing.assert_close(a["lora"][k], b["lora"][k], atol=1e-6, rtol=0)
