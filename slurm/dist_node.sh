#!/usr/bin/env zsh
# Per-node worker for sbatch_train_dist.sh (run once per node via srun).
#
# Starts this node's vLLM on GPUs 0..VLLM_TP_SIZE-1, waits for EVERY node's
# vLLM /health, then runs torchrun with TRAINER_GPUS_PER_NODE ranks on the
# remaining GPUs. Env comes from sbatch_train_dist.sh; MASTER_ADDR, VLLM_URLS
# etc. are recomputed from SLURM_JOB_NODELIST if missing.
#
# DRY_RUN=1 prints this node's topology and torchrun command and exits.

set -e

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/ksnair/worktrees/slurm}}"
cd "$REPO_DIR"

: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST must be set}"
: "${SLURM_NODEID:?SLURM_NODEID must be set (run via srun)}"
: "${LORA_RANK:=32}"
: "${GROUP_SIZE:=16}"
: "${GROUPS_PER_BATCH:=128}"
: "${LEARNING_RATE:=7e-5}"
: "${ENV:=mixed}"
: "${DATASET_SCHEDULE:=m-m}"
: "${LOSS_FN:=ppo}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"  # vLLM has its own GPU(s) here
: "${VLLM_TP_SIZE:=1}"
: "${GPUS_PER_NODE:=${SLURM_GPUS_ON_NODE:-4}}"
: "${TRAINER_GPUS_PER_NODE:=$(( GPUS_PER_NODE - VLLM_TP_SIZE ))}"
: "${EXTRA_ARGS:=}"
: "${DRY_RUN:=0}"
: "${LOG_DIR:=/data/hf_cache/ksnair/CuriousLLMs_logs/${SLURM_JOB_ID}-$(echo "$MODEL_NAME" | tr '/' '-')}"

HOSTS=(${(f)"$(scontrol show hostnames "$SLURM_JOB_NODELIST")"})
: "${SLURM_NNODES:=${#HOSTS[@]}}"
: "${MASTER_ADDR:=${HOSTS[1]}}"
: "${MASTER_PORT:=$(( 29500 + SLURM_JOB_ID % 1000 ))}"
: "${VLLM_PORT:=$(( 20000 + SLURM_JOB_ID % 9000 ))}"
if [[ -z "${VLLM_URLS}" ]]; then
    URLS=(); for h in "${HOSTS[@]}"; do URLS+=("http://${h}:${VLLM_PORT}"); done
    VLLM_URLS=${(j:,:)URLS}
fi
export VLLM_PORT LOG_DIR VLLM_TP_SIZE VLLM_GPU_MEM_FRAC MODEL_NAME LORA_RANK

if (( TRAINER_GPUS_PER_NODE < 1 || VLLM_TP_SIZE + TRAINER_GPUS_PER_NODE > GPUS_PER_NODE )); then
    echo "[dist-node] bad GPU split: vLLM=$VLLM_TP_SIZE trainer=$TRAINER_GPUS_PER_NODE total=$GPUS_PER_NODE" >&2
    exit 1
fi
VLLM_IDS=(${(f)"$(seq 0 $((VLLM_TP_SIZE-1)))"})
TRAINER_IDS=(${(f)"$(seq $VLLM_TP_SIZE $((GPUS_PER_NODE-1)))"})
VLLM_VISIBLE=${(j:,:)VLLM_IDS}
TRAINER_VISIBLE=${(j:,:)TRAINER_IDS}

TORCHRUN_CMD=(torchrun
    --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID"
    --nproc_per_node="$TRAINER_GPUS_PER_NODE"
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
    --rdzv_id="$SLURM_JOB_ID"
    math_train.py
    backend=local
    model_name="$MODEL_NAME"
    lora_rank="$LORA_RANK"
    group_size="$GROUP_SIZE"
    groups_per_batch="$GROUPS_PER_BATCH"
    learning_rate="$LEARNING_RATE"
    env="$ENV"
    dataset_schedule="$DATASET_SCHEDULE"
    loss_fn="$LOSS_FN"
    base_url="$VLLM_URLS"
    log_path="$LOG_DIR"
    behavior_if_log_dir_exists=resume)
[[ -n "${INIT_FROM_ADAPTER}" ]] && TORCHRUN_CMD+=(init_from_adapter="$INIT_FROM_ADAPTER")
TORCHRUN_CMD+=(${=EXTRA_ARGS})

echo "[dist-node $SLURM_NODEID/$SLURM_NNODES $(hostname -s)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT VLLM_PORT=$VLLM_PORT"
echo "[dist-node $SLURM_NODEID] VLLM_URLS=$VLLM_URLS"
echo "[dist-node $SLURM_NODEID] vLLM CUDA_VISIBLE_DEVICES=$VLLM_VISIBLE  trainer CUDA_VISIBLE_DEVICES=$TRAINER_VISIBLE (LOCAL_TRAINER_GPU_MEM_FRAC=1.0)"
echo "[dist-node $SLURM_NODEID] torchrun: ${TORCHRUN_CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

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
# Node-local Triton cache for vLLM AND the trainer (~/.triton may point at a
# /scratch dir missing on this node). Set here in the parent scope because
# launch_vllm.sh (same default) may run in a subshell whose exports are lost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER}-${SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"

mkdir -p "$LOG_DIR"
# Per-node vLLM log so nodes sharing LOG_DIR do not interleave.
VLLM_LOG_DIR="$LOG_DIR/node${SLURM_NODEID}"
mkdir -p "$VLLM_LOG_DIR"

# --- vLLM in a subshell with restricted CUDA_VISIBLE_DEVICES -------
(
    export CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE"
    export LOG_DIR="$VLLM_LOG_DIR"
    source slurm/launch_vllm.sh
    wait $VLLM_PID
) &
VLLM_BG_PID=$!
stop_vllm() {
    kill "$VLLM_BG_PID" 2>/dev/null
    pkill -P "$VLLM_BG_PID" 2>/dev/null
}
on_term() {
    echo "[dist-node $SLURM_NODEID] SIGTERM (preempted?) - forwarding to torchrun and vLLM" >&2
    set +e  # errexit must not abort the handler (wait returns 143 after the kill)
    [[ -n "$TRAIN_PID" ]] && kill -TERM "$TRAIN_PID" 2>/dev/null
    [[ -n "$TRAIN_PID" ]] && wait "$TRAIN_PID" 2>/dev/null
    stop_vllm
    exit 143
}
trap stop_vllm EXIT INT
trap on_term TERM

# Wait for every node's vLLM (rank 0 samples from all of them).
for url in ${(s:,:)VLLM_URLS}; do
    echo "[dist-node $SLURM_NODEID] waiting on $url/health ..."
    attempts=0
    until curl -sf "$url/health" > /dev/null; do
        if ! kill -0 "$VLLM_BG_PID" 2>/dev/null; then
            echo "[dist-node $SLURM_NODEID] local vLLM exited prematurely" >&2
            tail -200 "$VLLM_LOG_DIR/vllm.log" >&2 || true
            exit 1
        fi
        sleep 5
        attempts=$((attempts + 1))
        if (( attempts > 240 )); then
            echo "[dist-node $SLURM_NODEID] $url not ready after 20 min" >&2
            tail -200 "$VLLM_LOG_DIR/vllm.log" >&2 || true
            exit 1
        fi
    done
done

# --- trainer on the GPUs vLLM is not using -------------------------
export CUDA_VISIBLE_DEVICES="$TRAINER_VISIBLE"
export LOCAL_TRAINER_GPU_MEM_FRAC=1.0

# Background + wait so the TERM trap can forward preemption to torchrun.
"${TORCHRUN_CMD[@]}" &
TRAIN_PID=$!
wait $TRAIN_PID
