#!/usr/bin/env zsh
# Per-node worker for sbatch_train_dist.sh (run once per node via srun).
#
# On Modal (CLLM_PLATFORM=modal) sbatch_train_dist.sh execs this in every container; the node
# index is CLLM_NODE_RANK instead of SLURM_NODEID (see slurm/platform.sh).
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
source "$REPO_DIR/slurm/platform.sh"   # Babel vs Modal paths, envs, hosts, signals

: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${JOB_NUM:?no job id (SLURM_JOB_ID / JOB_ID)}"
NODE_ID=${SLURM_NODEID:-${CLLM_NODE_RANK:-}}
: "${NODE_ID:?SLURM_NODEID must be set (run via srun), or CLLM_NODE_RANK on Modal}"
: "${LORA_RANK:=32}"
: "${GROUP_SIZE:=16}"
: "${GROUPS_PER_BATCH:=128}"
: "${LEARNING_RATE:=7e-5}"
: "${ENV:=mixed}"
: "${DATASET_SCHEDULE:=m-m}"
: "${LOSS_FN:=ppo}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"  # vLLM has its own GPU(s) here
: "${VLLM_TP_SIZE:=1}"
: "${GPUS_PER_NODE:=$(cllm_gpus_on_node)}"
: "${GPUS_PER_NODE:=4}"
: "${TRAINER_GPUS_PER_NODE:=$(( GPUS_PER_NODE - VLLM_TP_SIZE ))}"
: "${EXTRA_ARGS:=}"
: "${DRY_RUN:=0}"
: "${LOG_DIR:=${LOGS_ROOT}/${JOB_ID}-$(echo "$MODEL_NAME" | tr '/' '-')}"

HOSTS=(${(f)"$(cllm_hosts)"})
NNODES=${SLURM_NNODES:-${#HOSTS[@]}}
: "${MASTER_ADDR:=${HOSTS[1]}}"
: "${MASTER_PORT:=$(( 29500 + JOB_NUM % 1000 ))}"
: "${VLLM_PORT:=$(( 20000 + JOB_NUM % 9000 ))}"
if [[ -z "${VLLM_URLS}" ]]; then
    URLS=(); for h in "${HOSTS[@]}"; do URLS+=("http://$(cllm_url_host "$h"):${VLLM_PORT}"); done
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
    --nnodes="$NNODES" --node_rank="$NODE_ID"
    --nproc_per_node="$TRAINER_GPUS_PER_NODE"
    --rdzv_backend=c10d --rdzv_endpoint="$(cllm_url_host "$MASTER_ADDR"):$MASTER_PORT"
    --rdzv_id="$JOB_ID"
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

echo "[dist-node $NODE_ID/$NNODES $(hostname -s)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT VLLM_PORT=$VLLM_PORT"
echo "[dist-node $NODE_ID] VLLM_URLS=$VLLM_URLS"
echo "[dist-node $NODE_ID] vLLM CUDA_VISIBLE_DEVICES=$VLLM_VISIBLE  trainer CUDA_VISIBLE_DEVICES=$TRAINER_VISIBLE (LOCAL_TRAINER_GPU_MEM_FRAC=1.0)"
echo "[dist-node $NODE_ID] torchrun: ${TORCHRUN_CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

cllm_activate clm
cllm_hf_cache shared
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
# Node-local Triton cache for vLLM AND the trainer (~/.triton may point at a
# /scratch dir missing on this node). Set here in the parent scope because
# launch_vllm.sh (same default) may run in a subshell whose exports are lost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER}-${JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"

mkdir -p "$LOG_DIR"
# Per-node vLLM log so nodes sharing LOG_DIR do not interleave.
VLLM_LOG_DIR="$LOG_DIR/node${NODE_ID}"
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
    echo "[dist-node $NODE_ID] SIGTERM (preempted?) - forwarding to torchrun and vLLM" >&2
    set +e  # errexit must not abort the handler (wait returns 143 after the kill)
    [[ -n "$TRAIN_PID" ]] && kill -TERM "$TRAIN_PID" 2>/dev/null
    [[ -n "$TRAIN_PID" ]] && wait "$TRAIN_PID" 2>/dev/null
    stop_vllm
    exit 143
}
trap stop_vllm EXIT INT
cllm_trap_preempt on_term

# Wait for every node's vLLM (rank 0 samples from all of them).
for url in ${(s:,:)VLLM_URLS}; do
    echo "[dist-node $NODE_ID] waiting on $url/health ..."
    attempts=0
    until curl -sf "$url/health" > /dev/null; do
        if ! kill -0 "$VLLM_BG_PID" 2>/dev/null; then
            echo "[dist-node $NODE_ID] local vLLM exited prematurely" >&2
            tail -200 "$VLLM_LOG_DIR/vllm.log" >&2 || true
            exit 1
        fi
        sleep 5
        attempts=$((attempts + 1))
        if (( attempts > 240 )); then
            echo "[dist-node $NODE_ID] $url not ready after 20 min" >&2
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
