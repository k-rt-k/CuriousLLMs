"""Run the Babel job scripts (slurm/*.sh) unchanged on Modal.

Each job runs the same zsh script it runs under sbatch, with CLLM_PLATFORM=modal; every Babel/Modal
difference (Python envs, caches, LOG_DIR root, hosts, preemption signals) lives in
slurm/platform.sh. The image replicates Babel's two micromamba envs as venvs (/opt/envs/clm,
/opt/envs/clm-vllm) from modal_jobs/requirements-*.txt; the repo is mounted at /root/repo at
container start (edits need no image rebuild).

    # once: data volume, secrets, RND embeddings (see slurm/README.md "Modal")
    modal run --detach modal_jobs/app.py --script lean \
        --vars "MODEL_NAME=Qwen/Qwen3-8B NUM_BATCHES=3 EVAL_EVERY=0"

  --script    lean | format | eval | train | train_bigmodel | train_dist  (slurm/sbatch_<script>.sh)
  --vars      "KEY=VALUE ..." exported to the script, as `sbatch --export=ALL,KEY=VALUE` would
  --run-id    JOB_ID (default <script>-<UTC timestamp>); LOG_DIR is /vol/logs/<run-id>-<model>.
              Pass an earlier run's id to resume it. Modal's own preemption retries reuse it.

Resources are read when `modal run` imports this file (and baked into the image env so the
container sees the same values):
  CLLM_MODAL_GPU      GPU spec per node (default "L40S:4"); e.g. "A100-80GB:4", "H100:8".
                      Empty: CPU-only (DRY_RUN=1 checks, check_image without GPU billing)
  CLLM_MODAL_NODES    nodes (default 1). >1 uses a Modal cluster: full nodes only (e.g. "H100:8"),
                      every container runs the script with its node index (lean, train_dist).
  CLLM_MODAL_RDMA     1 for RDMA between nodes (H100/H200/B200 class; default 0)
  CLLM_MODAL_RETRIES  retries after a failure or the 24 h timeout (default 2). Preemptions are
                      retried by Modal regardless; each retry resumes from LOG_DIR.
  CLLM_MODAL_SECRETS  comma list of Modal secrets (default "cllm-env": GEMINI/WANDB/TINKER keys)

Volume "cllm-data" (v2) at /vol: logs/ (LOG_DIRs and <run-id>.node<N>.out job logs), hf/ (HF
cache), cache/ (tiktoken etc.), embeddings/embeddings_gte_10000.pkl (RND normaliser).
"""

from __future__ import annotations

import datetime
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/repo"
HERE = Path(__file__).resolve().parent

GPU = os.environ.get("CLLM_MODAL_GPU", "L40S:4") or None
NODES = int(os.environ.get("CLLM_MODAL_NODES", "1"))
RDMA = os.environ.get("CLLM_MODAL_RDMA", "0") == "1"
RETRIES = int(os.environ.get("CLLM_MODAL_RETRIES", "2"))
SECRET_NAMES = [s for s in os.environ.get("CLLM_MODAL_SECRETS", "cllm-env").split(",") if s]

SCRIPTS = {
    "lean": "slurm/sbatch_lean.sh",
    "format": "slurm/sbatch_format.sh",
    "eval": "slurm/sbatch_eval.sh",
    "train": "slurm/sbatch_train.sh",
    "train_bigmodel": "slurm/sbatch_train_bigmodel.sh",
    "train_dist": "slurm/sbatch_train_dist.sh",
}
MULTI_NODE_SCRIPTS = {"lean", "train_dist"}

VOL = modal.Volume.from_name("cllm-data", create_if_missing=True, version=2)
RND_EMBEDDINGS = "embeddings/embeddings_gte_10000.pkl"

_UV = "uv pip install --no-deps --python"
image = (
    modal.Image.debian_slim(python_version="3.11")
    # zsh: the scripts; build-essential: host compiler for FlashInfer's nvcc JIT; procps: pkill
    .apt_install("zsh", "curl", "git", "procps", "build-essential")
    .pip_install("uv")
    .add_local_file(HERE / "requirements-clm.txt", "/opt/req/clm.txt", copy=True)
    .add_local_file(HERE / "requirements-clm-vllm.txt", "/opt/req/clm-vllm.txt", copy=True)
    .run_commands(
        "uv venv --python 3.11 /opt/envs/clm",
        f"{_UV} /opt/envs/clm/bin/python -r /opt/req/clm.txt"
        " --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match",
        "uv venv --python 3.11 /opt/envs/clm-vllm",
        f"{_UV} /opt/envs/clm-vllm/bin/python -r /opt/req/clm-vllm.txt",
    )
    .add_local_dir(REPO / "lean_rl" / "vllm_plugin", "/opt/vllm_plugin", copy=True)
    .run_commands(
        f"{_UV} /opt/envs/clm-vllm/bin/python /opt/vllm_plugin",
        f"{_UV} /opt/envs/clm/bin/python /opt/vllm_plugin",
        # Same fix as on Babel: FlashInfer's JIT looks for lib64 and an unversioned libcudart.so
        # in the pip CUDA toolkit that sbatch_lean.sh makes CUDA_HOME.
        "d=$(echo /opt/envs/clm-vllm/lib/python3.11/site-packages/nvidia/cu13) && "
        "ln -sfn lib $d/lib64 && "
        "{ [ -e $d/lib/libcudart.so ] || [ ! -e $d/lib/libcudart.so.13 ] || ln -s libcudart.so.13 $d/lib/libcudart.so; }",
    )
    .env({
        "CLLM_MODAL_GPU": GPU or "",
        "CLLM_MODAL_NODES": str(NODES),
        "CLLM_MODAL_RDMA": "1" if RDMA else "0",
        "CLLM_MODAL_RETRIES": str(RETRIES),
        "CLLM_MODAL_SECRETS": ",".join(SECRET_NAMES),
    })
    .add_local_dir(
        REPO,
        REMOTE_REPO,
        ignore=[
            ".git", ".env", ".claude", ".claude-flow", ".swarm", ".superpowers", ".pytest_cache",
            "logs", "notebooks", "papers", "batch_data", "RND_experimentation",
            "embeddings", "**/__pycache__", "**/*.ipynb",
        ],
    )
)
if RDMA:
    image = image.apt_install("libibverbs1")

app = modal.App("cllm-jobs", image=image)
N_GPUS = 0 if GPU is None else int(GPU.rsplit(":", 1)[1]) if ":" in GPU else 1
FN = dict(
    gpu=GPU,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name(n) for n in SECRET_NAMES],
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=RETRIES, initial_delay=0.0),
    cpu=2.0 * max(N_GPUS, 1),           # as on Babel: 16 CPUs per 8 GPUs
    memory=32 * 1024 * max(N_GPUS, 1),  # MiB; gpt-oss FSDP rank 0 holds the full checkpoint in CPU RAM
)


@app.function(volumes={"/vol": VOL}, timeout=1800)
def check_image() -> None:
    """CPU-only: both envs import, the vLLM plugin is registered, zsh and the repo are present.
    `modal run modal_jobs/app.py::check_image`"""
    checks = [
        ["/opt/envs/clm/bin/python", "-c",
         "import torch, transformers, peft, tinker, chz, vllm; "
         "print('clm', torch.__version__, transformers.__version__, peft.__version__, vllm.__version__)"],
        ["/opt/envs/clm-vllm/bin/python", "-c",
         "import importlib.metadata as md, torch, vllm; "
         "eps=[e for e in md.entry_points(group='vllm.general_plugins') if e.name=='lean_rl_gpt_oss_lm_head']; "
         "assert eps and eps[0].load(); print('clm-vllm', torch.__version__, vllm.__version__, 'plugin ok')"],
        ["zsh", "-c", f"cd {REMOTE_REPO} && export CLLM_PLATFORM=modal JOB_ID=check && source slurm/platform.sh && "
         "cllm_activate clm && PYTHONPATH=./tinker-cookbook python -c 'import lean_rl.train, math_train; print(\"repo ok\")'"],
        ["sh", "-c", "ls /opt/envs/clm-vllm/lib/python3.11/site-packages/nvidia/cu13/bin/nvcc && which g++"],
    ]
    _prepare_volume()
    print("RND embeddings on volume:", (Path("/vol") / RND_EMBEDDINGS).exists())
    for cmd in checks:
        print("+", " ".join(cmd)[:160], flush=True)
        subprocess.run(cmd, check=True)


def _prepare_volume() -> None:
    for d in ("/vol/logs", "/vol/hf", "/vol/cache", "/vol/embeddings"):
        os.makedirs(d, exist_ok=True)
    src, dst = Path("/vol") / RND_EMBEDDINGS, Path(REMOTE_REPO) / RND_EMBEDDINGS
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def _run(script: str, env: dict[str, str], run_id: str, node_rank: int, hosts: list[str]) -> int:
    """Run slurm/<script> as this node; stream its output to the Modal log and to
    /vol/logs/<run_id>.node<N>.out (appended across retries, like sbatch --open-mode=append)."""
    _prepare_volume()
    full_env = {
        **os.environ,
        **env,
        "CLLM_PLATFORM": "modal",
        "JOB_ID": run_id,
        "CLLM_HOSTS": ",".join(hosts),
        "CLLM_NODE_RANK": str(node_rank),
        "CLLM_SELF_HOST": hosts[node_rank],
        "REPO_DIR": REMOTE_REPO,
        "USER": os.environ.get("USER") or "root",
    }
    out_path = Path(f"/vol/logs/{run_id}.node{node_rank}.out")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    proc = subprocess.Popen(
        ["zsh", f"{REMOTE_REPO}/{SCRIPTS[script]}"],
        cwd=REMOTE_REPO, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    forwarded = threading.Event()

    def forward(signum, frame):  # Modal's preemption notice: let the script's ranks save
        if not forwarded.is_set():
            forwarded.set()
            print(f"[modal_jobs] signal {signum}: forwarding SIGTERM to the job script", flush=True)
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, forward)
        except ValueError:  # not the main thread: the script still gets a process-group signal
            print(f"[modal_jobs] could not install a handler for signal {sig}", flush=True)

    with out_path.open("a") as out:
        out.write(f"\n===== {stamp} node {node_rank}/{len(hosts)} {script} {run_id} =====\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            out.write(line)
            out.flush()
        rc = proc.wait()
        out.write(f"===== exit {rc} =====\n")
    VOL.commit()
    return rc


def _parse_vars(vars: str) -> dict[str, str]:
    env = {}
    for tok in shlex.split(vars):
        key, sep, val = tok.partition("=")
        if not sep or not key:
            raise SystemExit(f"--vars: expected KEY=VALUE, got {tok!r}")
        env[key] = val
    return env


@app.function(**FN)
def run_job(script: str, env: dict[str, str], run_id: str) -> int:
    rc = _run(script, env, run_id, 0, ["127.0.0.1"])
    if rc != 0:
        raise RuntimeError(f"{SCRIPTS[script]} exited with {rc}")
    return rc


if NODES > 1:

    @app.function(**FN)
    @modal.experimental.clustered(size=NODES, rdma=RDMA)
    def run_job_cluster(script: str, env: dict[str, str], run_id: str) -> int:
        info = modal.experimental.get_cluster_info()
        rc = _run(script, env, run_id, info.rank, list(info.container_ipv4_ips))
        if rc != 0:
            raise RuntimeError(f"{SCRIPTS[script]} on node {info.rank} exited with {rc}")
        return rc


@app.local_entrypoint()
def main(script: str, vars: str = "", run_id: str = ""):
    if script not in SCRIPTS:
        raise SystemExit(f"--script must be one of {sorted(SCRIPTS)}")
    if NODES > 1 and script not in MULTI_NODE_SCRIPTS:
        raise SystemExit(f"CLLM_MODAL_NODES={NODES}: only {sorted(MULTI_NODE_SCRIPTS)} run multi-node")
    env = _parse_vars(vars)
    run_id = run_id or f"{script}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d-%H%M%S}"
    print(f"[modal_jobs] {script} run_id={run_id} gpu={GPU} nodes={NODES} vars={env}")
    print(f"[modal_jobs] logs: modal volume get cllm-data logs/{run_id}.node0.out ; LOG_DIR /vol/logs/{run_id}-*")
    fn = run_job_cluster if NODES > 1 else run_job
    rc = fn.remote(script, env, run_id)
    print(f"[modal_jobs] {script} {run_id}: exit {rc}")
