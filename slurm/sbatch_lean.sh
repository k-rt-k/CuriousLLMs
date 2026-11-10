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
#     GPUS_PER_NODE on every other node. One `srun` task per node re-executes this
#     script in --worker mode, which runs
#       torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=<n>
#                --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID
#                --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m lean_rl.train ...
#     Global rank 0 (any node) drives the loop and samples from node 0's vLLM.
#   - fsdp=True (EXTRA_ARGS or FSDP=1) shards the frozen base across all trainer ranks.
#
# Examples:
#   # Qwen3-8B, 1 node, replicated trainer (6 trainer ranks, vLLM on GPU 0):
#   sbatch --gres=gpu:L40S:8 --export=ALL,MODEL_NAME=Qwen/Qwen3-8B slurm/sbatch_lean.sh
#   # gpt-oss-20b on 80 GB+ cards, replicated, lm_head plugin on:
#   sbatch --gres=gpu:A100_80GB:4 \
#          --export=ALL,MODEL_NAME=openai/gpt-oss-20b,GPT_OSS_LM_HEAD=1 slurm/sbatch_lean.sh
#   # gpt-oss-20b on 8x48 GB (FSDP shards the trainer; vLLM TP=1 on GPU 0):
#   sbatch --gres=gpu:L40S:8 \
#          --export=ALL,MODEL_NAME=openai/gpt-oss-20b,FSDP=1 slurm/sbatch_lean.sh
#   # 2 nodes:
#   sbatch --nodes=2 --export=ALL,MODEL_NAME=Qwen/Qwen3-8B slurm/sbatch_lean.sh
#
# Env knobs: MODEL_NAME LORA_RANK GROUP_SIZE GROUPS_PER_BATCH LEARNING_RATE ENV LOSS_FN
# VLLM_TP_SIZE VLLM_GPU_MEM_FRAC GPUS_PER_NODE FSDP(0|1) GPT_OSS_LM_HEAD(0|1) LOG_DIR
# EXTRA_ARGS ("key=value ..." forwarded to lean_rl.train). Optional pass-through when set:
# MICRO_BATCH_TOKENS NUM_BATCHES SAVE_EVERY KEEP_CHECKPOINTS EVAL_EVERY INIT_FROM_ADAPTER.
#
# Preemption: SIGTERM -> forwarded to srun -> each node's torchrun. The trainer saves a
# checkpoint at the next step boundary and exits 0; then vLLM is stopped. A requeued job
# (same SLURM_JOB_ID) reuses LOG_DIR and resumes.
#
# DRY_RUN=1 prints the topology and per-node commands and launches nothing.

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
: "${GPT_OSS_LM_HEAD:=0}"
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

# Per-node layout: sets NODE_NPROC and NODE_VISIBLE for node index $1 (0-based).
node_layout() {
    local n=$1
    if (( n == 0 )); then
        NODE_NPROC=$TRAINER_GPUS_NODE0
        NODE_VISIBLE=${(j:,:)${(f)"$(seq $VLLM_TP_SIZE $((GPUS_PER_NODE-1)))"}}
    else
        NODE_NPROC=$GPUS_PER_NODE
        NODE_VISIBLE=${(j:,:)${(f)"$(seq 0 $((GPUS_PER_NODE-1)))"}}
    fi
}

torchrun_cmd() {  # $1 = node_rank, $2 = nproc_per_node
    build_trainer_args
    TORCHRUN_CMD=(torchrun
        --nnodes="$NNODES" --node_rank="$1" --nproc_per_node="$2"
        --rdzv_backend=c10d --rdzv_id="$SLURM_JOB_ID"
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
        -m lean_rl.train "${TRAINER_ARGS[@]}")
}

if [[ "$1" != "--worker" ]]; then
    VLLM_VISIBLE=${(j:,:)${(f)"$(seq 0 $((VLLM_TP_SIZE-1)))"}}
    echo "[lean] nodes=$NNODES hosts=${HOSTS[*]}"
    echo "[lean] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT VLLM_PORT=$VLLM_PORT"
    echo "[lean] vllm_urls=$VLLM_URLS (node 0 only)  GPUS_PER_NODE=$GPUS_PER_NODE VLLM_TP_SIZE=$VLLM_TP_SIZE"
    echo "[lean] fsdp=$FSDP gpt_oss_lm_head_plugin=$USE_LM_HEAD_PLUGIN LOG_DIR=$LOG_DIR"
    echo "[lean] node 0: vLLM CUDA_VISIBLE_DEVICES=$VLLM_VISIBLE$([[ "$USE_LM_HEAD_PLUGIN" == 1 ]] && echo " VLLM_PLUGINS=lean_rl_gpt_oss_lm_head")"
    WORLD=0
    for (( i = 0; i < NNODES; i++ )); do
        node_layout $i
        torchrun_cmd $i $NODE_NPROC
        WORLD=$(( WORLD + NODE_NPROC ))
        echo "[lean] node $i (${HOSTS[$((i+1))]}): trainer ranks=$NODE_NPROC CUDA_VISIBLE_DEVICES=$NODE_VISIBLE"
        echo "[lean]   ${TORCHRUN_CMD[*]}"
    done
    echo "[lean] trainer world size=$WORLD"
    SRUN_CMD=(srun --ntasks-per-node=1 --kill-on-bad-exit=1 zsh "$REPO_DIR/slurm/sbatch_lean.sh" --worker)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[lean] DRY_RUN srun: ${SRUN_CMD[*]}"
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
# gpt-oss: HF modeling must use the eager attention path (sdpa/flash lack attention sinks).
[[ "${(L)MODEL_NAME}" == *gpt-oss* ]] && export LOCAL_ATTN_IMPL=eager
export LOCAL_TRAINER_GPU_MEM_FRAC=1.0   # trainer GPUs are not shared with vLLM

# ---- worker: one per node, runs the trainer ranks --------------------------
if [[ "$1" == "--worker" ]]; then
    node_layout "$SLURM_NODEID"
    torchrun_cmd "$SLURM_NODEID" "$NODE_NPROC"
    export CUDA_VISIBLE_DEVICES="$NODE_VISIBLE"
    echo "[lean-worker $SLURM_NODEID $(hostname -s)] CUDA_VISIBLE_DEVICES=$NODE_VISIBLE torchrun: ${TORCHRUN_CMD[*]}"
    on_term() {
        set +e
        echo "[lean-worker $SLURM_NODEID] SIGTERM - forwarding to torchrun (it saves a checkpoint and exits)" >&2
        [[ -n "$TRAIN_PID" ]] && kill -TERM "$TRAIN_PID" 2>/dev/null
        [[ -n "$TRAIN_PID" ]] && wait "$TRAIN_PID" 2>/dev/null
        exit $?
    }
    trap on_term TERM
    "${TORCHRUN_CMD[@]}" &
    TRAIN_PID=$!
    wait $TRAIN_PID
    exit $?
fi

# ---- parent (node 0): vLLM, then srun the workers ---------------------------
mkdir -p "$LOG_DIR"
echo "[lean] LOG_DIR=$LOG_DIR"

(
    export CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE"
    export VLLM_PORT VLLM_TP_SIZE VLLM_GPU_MEM_FRAC MODEL_NAME LORA_RANK LOG_DIR
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

# Preemption: forward TERM to the srun step (trainer saves + exits 0), wait, then stop vLLM.
on_term() {
    set +e  # errexit must not abort the handler (wait returns 143 after the kill)
    echo "[lean] SIGTERM (preempted?) - forwarding to the trainer step, then stopping vLLM" >&2
    [[ -n "$SRUN_PID" ]] && kill -TERM "$SRUN_PID" 2>/dev/null
    [[ -n "$SRUN_PID" ]] && wait "$SRUN_PID" 2>/dev/null
    rc=$?
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
    if (( attempts > 240 )); then
        echo "[lean] vLLM not ready after 20 min" >&2
        tail -200 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
done
echo "[lean] vLLM ready"

export MODEL_NAME LORA_RANK GROUP_SIZE GROUPS_PER_BATCH LEARNING_RATE ENV LOSS_FN VLLM_TP_SIZE
export GPUS_PER_NODE FSDP GPT_OSS_LM_HEAD EXTRA_ARGS USE_LM_HEAD_PLUGIN TRAINER_GPUS_NODE0
export MASTER_ADDR MASTER_PORT VLLM_PORT VLLM_URLS LOG_DIR REPO_DIR
"${SRUN_CMD[@]}" &
SRUN_PID=$!
rc=0
wait $SRUN_PID || rc=$?
stop_vllm
exit $rc
