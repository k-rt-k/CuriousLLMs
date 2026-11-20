#!/usr/bin/env zsh
#SBATCH --job-name=cllm-lean
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/home/ksnair/CuriousLLMs_logs/slurm-%x-%j.out
#
# Lean local RL trainer (lean_rl/, `python -m lean_rl.train`), 1..N nodes.
#
# Topology (see docs/pipeline.md "Lean local trainer"):
#   - Node 0 (the batch host) runs the ONLY vLLM, on GPUs 0..VLLM_TP_SIZE-1.
#   - Trainer ranks: GPUS_PER_NODE - VLLM_TP_SIZE on node 0 (GPUs VLLM_TP_SIZE..end),
#     GPUS_PER_NODE on every other node. NO torchrun: one `srun` task per trainer rank
#     (SLURM_HOSTFILE + --distribution=arbitrary lists node 0 G-TP times, others G times);
#     each task runs this script in --rank mode, which exports the torchrun-style env
#       RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS MASTER_ADDR/MASTER_PORT LOCAL_RANK LOCAL_WORLD_SIZE
#     and execs `python -m lean_rl.train ...` (lean_rl.dist.init_dist reads that env).
#     LOCAL_RANK is the PHYSICAL GPU index (node 0: TP..G-1, others 0..G-1) and every GPU of
#     the node stays visible (CUDA_VISIBLE_DEVICES=0..G-1), so cuda:LOCAL_RANK is the rank's GPU.
#     Global rank 0 (node 0) drives the loop and samples from node 0's vLLM.
#   - FSDP=1 (-> fsdp=True; do NOT also put fsdp= in EXTRA_ARGS) shards the frozen base across
#     all trainer ranks.
#   - preempt_qos caps a user at 24 GPUs total; --nodes x --gres must fit under that.
#   - Attention implementation is chosen by the trainer (eager for gpt-oss, sdpa otherwise);
#     override with EXTRA_ARGS="attn_implementation=...".
#
# Examples:
#   # Qwen3-8B, 1 node, replicated trainer (7 trainer ranks, vLLM on GPU 0):
#   sbatch --gres=gpu:L40S:8 --export=ALL,MODEL_NAME=Qwen/Qwen3-8B slurm/sbatch_lean.sh
#   # gpt-oss-20b on 80 GB+ cards, replicated (lm_head plugin on by default, GPT_OSS_LM_HEAD=1):
#   sbatch --gres=gpu:A100_80GB:4 \
#          --export=ALL,MODEL_NAME=openai/gpt-oss-20b slurm/sbatch_lean.sh
#   # gpt-oss-20b on 8x48 GB (FSDP shards the trainer; vLLM TP=1 on GPU 0):
#   sbatch --gres=gpu:L40S:8 \
#          --export=ALL,MODEL_NAME=openai/gpt-oss-20b,FSDP=1 slurm/sbatch_lean.sh
#   # 2 nodes:
#   sbatch --nodes=2 --export=ALL,MODEL_NAME=Qwen/Qwen3-8B slurm/sbatch_lean.sh
#
# Env knobs: MODEL_NAME LORA_RANK GROUP_SIZE GROUPS_PER_BATCH LEARNING_RATE ENV LOSS_FN
# MAX_TOKENS(2048) VLLM_TP_SIZE VLLM_GPU_MEM_FRAC VLLM_STARTUP_TIMEOUT_S(7200) VLLM_ENV(clm-vllm) GPUS_PER_NODE FSDP(0|1) GPT_OSS_LM_HEAD(1|0, default 1) LOG_DIR
# EXTRA_ARGS ("key=value ..." forwarded to lean_rl.train). Optional pass-through when set:
# MICRO_BATCH_TOKENS NUM_BATCHES SAVE_EVERY KEEP_CHECKPOINTS EVAL_EVERY INIT_FROM_ADAPTER.
#
# Preemption (Babel: GraceTime=0, KillWait=120 s, so the whole shutdown must fit in 120 s):
# every trainer rank is python exec'd directly (no torchelastic), so it gets Slurm's SIGTERM
# itself; the trainer's PreemptionFlag (SIGTERM/SIGUSR1) saves the last completed step and
# exits 0. On ONE node the ranks are children of this shell, not srun tasks: srun answers its
# own SIGTERM by SIGKILLing its tasks ("srun: forcing job termination"), which killed the ranks
# before they could save (job 11200771). The batch shell on TERM waits <= 100 s for the ranks,
# then stops vLLM and exits with the worst rank rc. Multi-node runs still need srun, so a
# preempted multi-node job loses the steps since its last periodic checkpoint (SAVE_EVERY).
# A requeued job (same SLURM_JOB_ID) reuses LOG_DIR and resumes.
#
# DRY_RUN=1 prints the hostfile, per-rank RANK/LOCAL_RANK/GPU mapping and commands; launches nothing.

set -e

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/ksnair/worktrees/slurm}}"
cd "$REPO_DIR"

: "${MODEL_NAME:=Qwen/Qwen3-8B}"
: "${LORA_RANK:=32}"
: "${GROUP_SIZE:=16}"
: "${GROUPS_PER_BATCH:=128}"
: "${LEARNING_RATE:=7e-5}"
: "${ENV:=mixed}"
: "${LOSS_FN:=ppo}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"  # vLLM has its own GPU(s)
: "${VLLM_TP_SIZE:=1}"
: "${GPUS_PER_NODE:=${SLURM_GPUS_ON_NODE:-8}}"
: "${FSDP:=0}"
: "${GPT_OSS_LM_HEAD:=1}"  # Tinker parity (lm_head LoRA); needs lean_rl/vllm_plugin installed
: "${MAX_TOKENS:=2048}"
: "${VLLM_STARTUP_TIMEOUT_S:=7200}"  # cold imports + /data weight reads on Babel have taken >40 min
: "${VLLM_ENV:=clm-vllm}"  # micromamba env for the vLLM server only (vLLM 0.33; the trainer always uses clm)
: "${EXTRA_ARGS:=}"
: "${DRY_RUN:=0}"

: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set (run under sbatch)}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST must be set (run under sbatch)}"

# Short hostnames resolve inside Babel; the same names serve MASTER_ADDR and vllm_urls.
HOSTS=(${(f)"$(scontrol show hostnames "$SLURM_JOB_NODELIST")"})
NNODES=${#HOSTS[@]}
MASTER_ADDR=${HOSTS[1]}
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
VLLM_PORT=$(( 20000 + SLURM_JOB_ID % 9000 ))
VLLM_URLS="http://${HOSTS[1]}:${VLLM_PORT}"   # only node 0 runs vLLM

if [[ -z "${LOG_DIR}" ]]; then
    # No timestamp: a requeued job (same SLURM_JOB_ID) resumes into the same LOG_DIR.
    LOG_DIR=/data/hf_cache/ksnair/CuriousLLMs_logs/${SLURM_JOB_ID}-$(echo "$MODEL_NAME" | tr '/' '-')
fi

TRAINER_GPUS_NODE0=$(( GPUS_PER_NODE - VLLM_TP_SIZE ))
if (( TRAINER_GPUS_NODE0 < 1 )); then
    echo "[lean] no trainer GPUs on node 0: GPUS_PER_NODE=$GPUS_PER_NODE VLLM_TP_SIZE=$VLLM_TP_SIZE" >&2
    exit 1
fi

# Plugin only for gpt-oss + GPT_OSS_LM_HEAD=1 (then the trainer also trains lm_head LoRA).
USE_LM_HEAD_PLUGIN=0
if [[ "${(L)MODEL_NAME}" == *gpt-oss* && "$GPT_OSS_LM_HEAD" == "1" ]]; then
    USE_LM_HEAD_PLUGIN=1
fi

# Trainer CLI args (chz key=value); same function runs in the parent and the worker.
build_trainer_args() {
    TRAINER_ARGS=(
        model_name="$MODEL_NAME"
        env="$ENV"
        lora_rank="$LORA_RANK"
        group_size="$GROUP_SIZE"
        groups_per_batch="$GROUPS_PER_BATCH"
        learning_rate="$LEARNING_RATE"
        max_tokens="$MAX_TOKENS"
        loss_fn="$LOSS_FN"
        vllm_urls="$VLLM_URLS"
        log_path="$LOG_DIR"
    )
    [[ "$FSDP" == "1" ]] && TRAINER_ARGS+=(fsdp=True)
    [[ "$USE_LM_HEAD_PLUGIN" == "1" ]] && TRAINER_ARGS+=(gpt_oss_lm_head=True)
    [[ -n "${MICRO_BATCH_TOKENS}" ]] && TRAINER_ARGS+=(micro_batch_tokens="$MICRO_BATCH_TOKENS")
    [[ -n "${NUM_BATCHES}" ]] && TRAINER_ARGS+=(num_batches="$NUM_BATCHES")
    [[ -n "${SAVE_EVERY}" ]] && TRAINER_ARGS+=(save_every="$SAVE_EVERY")
    [[ -n "${KEEP_CHECKPOINTS}" ]] && TRAINER_ARGS+=(keep_checkpoints="$KEEP_CHECKPOINTS")
    [[ -n "${EVAL_EVERY}" ]] && TRAINER_ARGS+=(eval_every="$EVAL_EVERY")
    [[ -n "${INIT_FROM_ADAPTER}" ]] && TRAINER_ARGS+=(init_from_adapter="$INIT_FROM_ADAPTER")
    TRAINER_ARGS+=(${=EXTRA_ARGS})
}

# Rank layout, in global-rank order (hostfile order). Zsh arrays are 1-indexed: rank r is [r+1].
RANK_HOST=(); RANK_NODE=(); RANK_GPU=()
for (( n = 0; n < NNODES; n++ )); do
    first=$(( n == 0 ? VLLM_TP_SIZE : 0 ))
    for (( g = first; g < GPUS_PER_NODE; g++ )); do
        RANK_HOST+=(${HOSTS[$((n+1))]}); RANK_NODE+=($n); RANK_GPU+=($g)
    done
done
WORLD=${#RANK_HOST[@]}
HOSTFILE="${TMPDIR:-/tmp}/lean-hostfile-${SLURM_JOB_ID}"
CPUS_PER_RANK=${CPUS_PER_RANK:-2}   # 8 ranks x 2 = the 16 CPUs the header requests per node

if [[ "$1" != "--rank" ]]; then
    VLLM_VISIBLE=${(j:,:)${(f)"$(seq 0 $((VLLM_TP_SIZE-1)))"}}
    echo "[lean] nodes=$NNODES hosts=${HOSTS[*]}"
    echo "[lean] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT VLLM_PORT=$VLLM_PORT"
    echo "[lean] vllm_urls=$VLLM_URLS (node 0 only)  GPUS_PER_NODE=$GPUS_PER_NODE VLLM_TP_SIZE=$VLLM_TP_SIZE"
    echo "[lean] fsdp=$FSDP gpt_oss_lm_head_plugin=$USE_LM_HEAD_PLUGIN LOG_DIR=$LOG_DIR"
    echo "[lean] node 0: vLLM (env $VLLM_ENV) CUDA_VISIBLE_DEVICES=$VLLM_VISIBLE$([[ "$USE_LM_HEAD_PLUGIN" == 1 ]] && echo " VLLM_PLUGINS=lean_rl_gpt_oss_lm_head")"
    echo "[lean] trainer world size=$WORLD; hostfile $HOSTFILE:"
    printf '[lean]   %s\n' "${RANK_HOST[@]}"
    for (( r = 0; r < WORLD; r++ )); do
        echo "[lean] rank $r: host=${RANK_HOST[$((r+1))]} node=${RANK_NODE[$((r+1))]} LOCAL_RANK=${RANK_GPU[$((r+1))]} (cuda:${RANK_GPU[$((r+1))]})"
    done
    build_trainer_args
    echo "[lean] per-rank command: RANK=<r> WORLD_SIZE=$WORLD LOCAL_RANK=<gpu> MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT python -m lean_rl.train ${TRAINER_ARGS[*]}"
    SRUN_CMD=(srun --distribution=arbitrary --ntasks="$WORLD" --kill-on-bad-exit=1 --overlap
        --gres=gpu:"$GPUS_PER_NODE" --gpu-bind=none --cpus-per-task="$CPUS_PER_RANK"
        zsh "$REPO_DIR/slurm/sbatch_lean.sh" --rank)
    if [[ "$DRY_RUN" == "1" ]]; then
        if (( NNODES == 1 )); then
            echo "[lean] DRY_RUN single node: $WORLD local rank processes: zsh $REPO_DIR/slurm/sbatch_lean.sh --rank"
        else
            echo "[lean] DRY_RUN SLURM_HOSTFILE=$HOSTFILE srun: ${SRUN_CMD[*]}"
        fi
        exit 0
    fi
fi

# ---- environment (parent and workers) --------------------------------------
export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate clm

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
unset HF_HOME
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
# Node-local Triton cache for vLLM AND the trainer (~/.triton may point at a /scratch
# dir missing on this node). Exported in the parent scope because launch_vllm.sh runs
# in a subshell whose exports are lost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER}-${SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"
# openai_harmony (gpt-oss tokenizer in vLLM) caches vocab files in /tmp/tiktoken-rs-cache by
# default; on shared nodes another user may own it (Permission denied -> vLLM dies at startup).
# Persistent per-user dir (a per-job $TMPDIR would re-download the vocab every job).
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_DIR:-/data/hf_cache/${USER}/tiktoken-rs}"
mkdir -p "$TIKTOKEN_RS_CACHE_DIR"

# ---- rank: one process per trainer rank (local child on 1 node, srun task otherwise) ----
if [[ "$1" == "--rank" ]]; then
    if [[ "$(hostname -s)" == "${HOSTS[1]}" ]]; then
        first=$VLLM_TP_SIZE; LOCAL_WORLD_SIZE=$TRAINER_GPUS_NODE0
    else
        first=0; LOCAL_WORLD_SIZE=$GPUS_PER_NODE
    fi
    export RANK=${LEAN_RANK:-$SLURM_PROCID} WORLD_SIZE=${LEAN_WORLD:-$SLURM_NTASKS}
    export LOCAL_RANK=$(( first + ${LEAN_LOCALID:-$SLURM_LOCALID} )) LOCAL_WORLD_SIZE
    export MASTER_ADDR MASTER_PORT
    export CUDA_VISIBLE_DEVICES=${(j:,:)${(f)"$(seq 0 $((GPUS_PER_NODE-1)))"}}
    build_trainer_args
    echo "[lean-rank $RANK/$WORLD_SIZE $(hostname -s)] LOCAL_RANK=$LOCAL_RANK (cuda:$LOCAL_RANK) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    # exec: python is the Slurm task itself and receives SIGTERM directly (PreemptionFlag).
    exec python -m lean_rl.train "${TRAINER_ARGS[@]}"
fi

# ---- parent (node 0): vLLM, then srun the ranks ---------------------------
# Checks the vllm.general_plugins entry point is registered (and importable); stderr kept so a
# broken install is not misreported as missing. Not run under DRY_RUN.
if [[ "$USE_LM_HEAD_PLUGIN" == "1" ]] && ! "$MAMBA_ROOT_PREFIX/envs/$VLLM_ENV/bin/python" -c "
import importlib.metadata as md, sys
eps = [e for e in md.entry_points(group='vllm.general_plugins') if e.name == 'lean_rl_gpt_oss_lm_head']
sys.exit(0 if eps and eps[0].load() else 1)"; then
    echo "[lean] gpt-oss with GPT_OSS_LM_HEAD=1 (default) needs the vLLM lm_head plugin:" >&2
    echo "[lean]   (in env $VLLM_ENV) python -m pip install --no-deps -e lean_rl/vllm_plugin" >&2
    echo "[lean] or set GPT_OSS_LM_HEAD=0 to train without lm_head LoRA (documented Tinker mismatch)." >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
echo "[lean] LOG_DIR=$LOG_DIR"

(
    [[ "$VLLM_ENV" != clm ]] && micromamba activate "$VLLM_ENV"
    # FlashInfer JIT-compiles kernels (sampling, gpt-oss attention) and many Babel nodes have
    # no /usr/local/cuda/bin/nvcc: use the env's pip CUDA toolkit (see slurm/README.md).
    for d in "$CONDA_PREFIX"/lib/python3.*/site-packages/nvidia/cu13; do
        [[ -x "$d/bin/nvcc" ]] && export CUDA_HOME="$d"
    done
    export CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE"
    export VLLM_PORT VLLM_TP_SIZE VLLM_GPU_MEM_FRAC MODEL_NAME LORA_RANK LOG_DIR VLLM_STARTUP_TIMEOUT_S
    [[ "$USE_LM_HEAD_PLUGIN" == "1" ]] && export VLLM_PLUGINS=lean_rl_gpt_oss_lm_head
    source slurm/launch_vllm.sh
    wait $VLLM_PID
) &
VLLM_BG_PID=$!
stop_vllm() {
    kill "$VLLM_BG_PID" 2>/dev/null
    pkill -P "$VLLM_BG_PID" 2>/dev/null
}
trap stop_vllm EXIT INT

# Preemption: ranks get Slurm's SIGTERM themselves (save + exit 0). Single node: TERM them again
# (harmless; covers a TERM sent to this shell alone), wait <= 100 s, then stop vLLM. Multi-node:
# forward TERM to srun (which SIGKILLs its tasks; see "Preemption" above).
RANK_PIDS=()
any_rank_alive() {
    local p
    for p in "${RANK_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done
    return 1
}
on_term() {
    set +e  # errexit must not abort the handler (wait returns 143 after the kill)
    echo "[lean] SIGTERM (preempted?) - waiting for the trainer ranks, then stopping vLLM" >&2
    if (( ${#RANK_PIDS} )); then
        kill -TERM "${RANK_PIDS[@]}" 2>/dev/null
        local i p s
        for (( i = 0; i < 100; i++ )); do   # KillWait is 120 s; leave room to stop vLLM
            any_rank_alive || break
            sleep 1
        done
        kill -KILL "${RANK_PIDS[@]}" 2>/dev/null
        rc=0
        for p in "${RANK_PIDS[@]}"; do
            wait "$p" 2>/dev/null; s=$?
            (( s != 0 && rc == 0 )) && rc=$s
        done
    elif [[ -n "$SRUN_PID" ]]; then
        kill -TERM "$SRUN_PID" 2>/dev/null
        local i
        for (( i = 0; i < 100; i++ )); do   # KillWait is 120 s; leave room to stop vLLM
            kill -0 "$SRUN_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$SRUN_PID" 2>/dev/null && kill -KILL "$SRUN_PID" 2>/dev/null
        wait "$SRUN_PID" 2>/dev/null
        rc=$?
    else
        rc=0
    fi
    stop_vllm
    exit $rc
}
trap on_term TERM

echo "[lean] waiting on http://127.0.0.1:${VLLM_PORT}/health ..."
attempts=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null; do
    if ! kill -0 "$VLLM_BG_PID" 2>/dev/null; then
        echo "[lean] vLLM exited prematurely" >&2
        tail -200 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
    sleep 5
    attempts=$((attempts + 1))
    if (( attempts * 5 > VLLM_STARTUP_TIMEOUT_S + 60 )); then   # launch_vllm.sh gives up first
        echo "[lean] vLLM not ready after ${VLLM_STARTUP_TIMEOUT_S}s" >&2
        tail -200 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
done
echo "[lean] vLLM ready"

export MODEL_NAME LORA_RANK GROUP_SIZE GROUPS_PER_BATCH LEARNING_RATE ENV LOSS_FN VLLM_TP_SIZE
export MAX_TOKENS GPUS_PER_NODE FSDP GPT_OSS_LM_HEAD EXTRA_ARGS USE_LM_HEAD_PLUGIN TRAINER_GPUS_NODE0
export MASTER_ADDR MASTER_PORT VLLM_PORT VLLM_URLS LOG_DIR REPO_DIR
if (( NNODES == 1 )); then
    # Local children (see "Preemption" above). Each rank's PID is python's (the --rank branch
    # execs it). Like srun --kill-on-bad-exit: once a rank fails, TERM the rest, KILL after 60 s.
    for (( r = 0; r < WORLD; r++ )); do
        LEAN_RANK=$r LEAN_WORLD=$WORLD LEAN_LOCALID=$r zsh "$REPO_DIR/slurm/sbatch_lean.sh" --rank &
        RANK_PIDS+=($!)
    done
    rc=0
    pending=("${RANK_PIDS[@]}")
    while (( ${#pending} )); do
        still=()
        for p in "${pending[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                still+=($p)
            else
                s=0; wait "$p" || s=$?
                (( s != 0 && rc == 0 )) && rc=$s
            fi
        done
        pending=("${still[@]}")
        if (( rc != 0 && ${#pending} )); then
            echo "[lean] a trainer rank exited with rc=$rc; stopping the others" >&2
            kill -TERM "${pending[@]}" 2>/dev/null || true   # some may have exited already
            for (( i = 0; i < 60; i++ )); do
                any_rank_alive || break
                sleep 1
            done
            kill -KILL "${pending[@]}" 2>/dev/null || true
            for p in "${pending[@]}"; do wait "$p" 2>/dev/null || true; done
            break
        fi
        (( ${#pending} )) && sleep 5
    done
    stop_vllm
    exit $rc
fi

printf '%s\n' "${RANK_HOST[@]}" > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"
# sbatch's header exports these job-level values, which srun reads as INPUT defaults and which
# would conflict with --ntasks/--distribution=arbitrary (e.g. ntasks-per-node=1); the explicit
# srun flags carry everything needed, and nothing later in this script uses them.
unset SLURM_NTASKS_PER_NODE SLURM_NTASKS SLURM_NNODES SLURM_TASKS_PER_NODE SLURM_CPUS_PER_TASK
"${SRUN_CMD[@]}" &
SRUN_PID=$!
rc=0
wait $SRUN_PID || rc=$?
stop_vllm
exit $rc
