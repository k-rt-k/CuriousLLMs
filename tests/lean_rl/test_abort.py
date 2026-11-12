"""Fail-fast abort across ranks: 2 real rank subprocesses (``tests.lean_rl.abort_loop``) with
LEAN_DIST_TIMEOUT_S=600, so finishing within 30 s of the failure proves the abort watchdog,
not the collective timeout. Clean and preempted runs must still exit 0 on both ranks."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("tinker")
pytest.importorskip("chz")

if os.environ.get("DIST_TEST_DEVICE", "cpu") != "cpu":
    pytest.skip("CPU (gloo) test", allow_module_level=True)

ABORT_WITHIN_S = 30.0
STARTUP_LIMIT_S = 600.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(model_dir, tmp_path, scenario, fsdp=False):
    """Start both ranks; return ({rank: returncode}, {rank: exit wall time}, {rank: log text})."""
    log = str(tmp_path / "run")
    port = _free_port()
    env = dict(os.environ, LEAN_DIST_TIMEOUT_S="600", PYTHONUNBUFFERED="1")
    procs, outs = {}, {}
    for rank in (0, 1):
        outs[rank] = open(tmp_path / f"rank{rank}.log", "w")
        procs[rank] = subprocess.Popen(
            [sys.executable, "-m", "tests.lean_rl.abort_loop", "--rank", str(rank),
             "--port", str(port), "--model", model_dir, "--log", log,
             "--scenario", scenario, "--fsdp", str(int(fsdp))],
            stdout=outs[rank], stderr=subprocess.STDOUT, env=env,
        )
    t0, ended = time.time(), {}
    try:
        while len(ended) < 2:
            for rank, p in procs.items():
                if rank not in ended and p.poll() is not None:
                    ended[rank] = time.time()
            fail_time = _fail_time(log)
            limit = fail_time + ABORT_WITHIN_S + 5 if fail_time else t0 + STARTUP_LIMIT_S
            if time.time() > limit:
                break
            time.sleep(0.2)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait()
        for f in outs.values():
            f.close()
    logs = {r: open(tmp_path / f"rank{r}.log").read() for r in (0, 1)}
    codes = {r: (p.returncode if r in ended else None) for r, p in procs.items()}
    return codes, ended, logs, _fail_time(log)


def _fail_time(log):
    path = os.path.join(log, "fail_time")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read().strip()
    return float(txt) if txt else None


def _assert_aborted(codes, ended, logs, fail_time, failing, msg):
    assert fail_time is not None, logs
    for r in (0, 1):
        assert codes[r] not in (None, 0), (r, codes, logs[r][-3000:])
        assert ended[r] - fail_time < ABORT_WITHIN_S, (r, ended[r] - fail_time)
    assert "Traceback" in logs[failing] and msg in logs[failing], logs[failing][-3000:]
    other = 1 - failing
    assert "aborting" in logs[other], logs[other][-3000:]


@pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])
def test_rank0_rollout_error_aborts_both(tiny_model_dir, tmp_path, fsdp):
    codes, ended, logs, ft = _launch(tiny_model_dir, tmp_path, "rollout_error", fsdp)
    _assert_aborted(codes, ended, logs, ft, failing=0, msg="boom in rollout")


@pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])
def test_rank1_train_step_error_aborts_both(tiny_model_dir, tmp_path, fsdp):
    codes, ended, logs, ft = _launch(tiny_model_dir, tmp_path, "train_error", fsdp)
    _assert_aborted(codes, ended, logs, ft, failing=1, msg="boom in train_step")


def test_rank1_setup_error_aborts_both(tiny_model_dir, tmp_path):
    codes, ended, logs, ft = _launch(tiny_model_dir, tmp_path, "comm_error")
    _assert_aborted(codes, ended, logs, ft, failing=1, msg="boom in setup")


class _ScriptedStore:
    """Fake store: each check() takes the next outcome ("err" raises, else returned); when the
    script runs out the watchdog is stopped."""

    def __init__(self, script, stop):
        self.script, self.stop, self.checks = list(script), stop, 0

    def check(self, keys):
        self.checks += 1
        out = self.script.pop(0)
        if not self.script:
            self.stop.set()
        if out == "err":
            raise ConnectionError("transient")
        return out

    def get(self, key):
        return b"1"


@pytest.mark.parametrize("script, expect", [
    (["err", False, "err", "err", False], None),  # never 3 consecutive failures
    (["err", "err", "err", False], "lost rank 0's c10d store (3 failed checks"),
    ([True, False], "aborting: rank 1 failed"),  # the key aborts at once
    (["err", "err", True, False], "aborting: rank 1 failed"),
])
def test_watchdog_store_failure_threshold(script, expect):
    import threading

    from lean_rl import dist as ldist

    stop = threading.Event()
    store = _ScriptedStore(script, stop)
    exits = []
    ldist._watch(store, stop, 0.0, exit_fn=exits.append)
    if expect is None:
        assert exits == [] and store.checks == len(script)
    else:
        assert len(exits) == 1 and expect in exits[0], exits
        assert store.checks == len(script) - 1  # exited at that check, not later


def test_hard_exit_does_not_wait_for_a_blocked_log_handler(monkeypatch):
    import logging
    import threading

    from lean_rl import dist as ldist

    class _Exit(Exception):
        pass

    def fake_exit(code):
        raise _Exit(code)

    blocked = threading.Event()

    class Stuck(logging.Handler):
        def emit(self, record):
            pass

        def flush(self):
            blocked.wait()  # a stalled NFS file / full pipe

    h = Stuck()
    logging.getLogger().addHandler(h)
    monkeypatch.setattr(ldist.os, "_exit", fake_exit)
    monkeypatch.setattr(ldist, "_FLUSH_TIMEOUT_S", 0.3)
    t0 = time.time()
    try:
        with pytest.raises(_Exit):
            ldist._hard_exit("aborting: test")
        assert time.time() - t0 < 2.0
    finally:
        logging.getLogger().removeHandler(h)
        blocked.set()


@pytest.mark.parametrize("failing", [0, 1])
def test_watchdog_ends_rank_blocked_outside_collectives(tiny_model_dir, tmp_path, failing):
    """The survivor's main thread never enters a collective (gloo would itself notice the dead
    peer; a hung NCCL op does not): only the watchdog can end it. failing=1: rank 0 sees the
    abort key; failing=0: rank 1 sees the key or rank 0's store vanish."""
    codes, ended, logs, ft = _launch(tiny_model_dir, tmp_path, f"watchdog_rank{failing}_fails")
    _assert_aborted(codes, ended, logs, ft, failing=failing, msg="boom outside any collective")
    other = 1 - failing
    assert f"[rank {other}] aborting: " in logs[other], logs[other][-3000:]
    if failing == 1:
        assert "aborting: rank 1 failed" in logs[0]


@pytest.mark.parametrize("scenario", ["ok", "preempt"])
def test_clean_and_preempted_runs_exit_zero(tiny_model_dir, tmp_path, scenario):
    codes, _, logs, ft = _launch(tiny_model_dir, tmp_path, scenario, fsdp=True)
    assert ft is None
    for r in (0, 1):
        assert codes[r] == 0, (r, codes, logs[r][-3000:])
        assert "aborting" not in logs[r] and f"rank {r} finished cleanly" in logs[r]
    ckpts = sorted(os.listdir(tmp_path / "run" / "ckpt"))
    assert ckpts == (["step_000001", "step_000002"] if scenario == "ok" else ["step_000001"])
