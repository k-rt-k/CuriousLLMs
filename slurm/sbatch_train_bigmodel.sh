#!/usr/bin/env zsh
#SBATCH --job-name=cllm-train-big
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/data/hf_cache/ksnair/CuriousLLMs_logs/slurm-%j.out
#
# Big-model RL training (gpt-oss-20b, Llama-3.1-8B-Instruct, etc.).
#
# Topology:
#   - vLLM uses GPUs 0..VLLM_TP_SIZE-1 with tensor parallelism
#     (default VLLM_TP_SIZE=2 -> GPUs 0,1).
#   - Trainer uses the remaining GPUs via CUDA_VISIBLE_DEVICES
#     (default: the LAST gpu = GPU 3 on a 4-GPU node).
#   - To put the trainer on multiple GPUs (FSDP), set TRAINER_NUM_GPUS>1 and
#     prefix math_train.py with `accelerate launch --num_processes=N`.
#
# Override defaults via the environment when sbatch'ing:
#   sbatch --export=ALL,MODEL_NAME=openai/gpt-oss-20b,LORA_RANK=16,VLLM_TP_SIZE=2 \
#          slurm/sbatch_train_bigmodel.sh

set -e

export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate clm

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
unset HF_HOME
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

: "${MODEL_NAME:=openai/gpt-oss-20b}"
: "${LORA_RANK:=16}"
: "${GROUP_SIZE:=8}"
: "${GROUPS_PER_BATCH:=64}"
: "${LEARNING_RATE:=5e-6}"
: "${ENV:=mixed}"
: "${DATASET_SCHEDULE:=m-m}"
: "${LOSS_FN:=ppo}"
: "${VLLM_TP_SIZE:=2}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"
: "${TRAINER_NUM_GPUS:=1}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR=/data/hf_cache/ksnair/CuriousLLMs_logs/${STAMP}-${SLURM_JOB_ID:-local}-$(echo "$MODEL_NAME" | tr '/' '-')
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[train-big] LOG_DIR=$LOG_DIR"

# Split visible GPUs: vLLM gets [0..VLLM_TP_SIZE-1], trainer gets the rest.
TOTAL_GPUS=$(nvidia-smi -L | wc -l)
if (( VLLM_TP_SIZE + TRAINER_NUM_GPUS > TOTAL_GPUS )); then
    echo "[train-big] not enough GPUs: vLLM=$VLLM_TP_SIZE + trainer=$TRAINER_NUM_GPUS > total=$TOTAL_GPUS" >&2
    exit 1
fi

# vLLM occupies GPUs 0..VLLM_TP_SIZE-1
VLLM_VISIBLE=$(seq -s, 0 $((VLLM_TP_SIZE-1)))
# Trainer occupies the next TRAINER_NUM_GPUS
TRAINER_VISIBLE=$(seq -s, $VLLM_TP_SIZE $((VLLM_TP_SIZE+TRAINER_NUM_GPUS-1)))
echo "[train-big] vLLM GPUs=$VLLM_VISIBLE  trainer GPUs=$TRAINER_VISIBLE"

cd /home/ksnair/worktrees/slurm
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"

# --- vLLM in a subshell with restricted CUDA_VISIBLE_DEVICES -------
(
    export CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE"
    export VLLM_TP_SIZE
    export VLLM_GPU_MEM_FRAC
    source slurm/launch_vllm.sh
    # Pause-forever: the parent will use VLLM_URL set into the parent env
    # via FIFO. Simpler: exec wait on the vllm process.
    wait $VLLM_PID
) &
VLLM_BG_PID=$!
trap "kill $VLLM_BG_PID 2>/dev/null; pkill -P $VLLM_BG_PID 2>/dev/null" EXIT INT TERM

# Wait for vLLM /health (the subshell's port is deterministic from SLURM_JOB_ID)
if [[ -n "${SLURM_JOB_ID}" ]]; then
    VLLM_PORT=$(( 20000 + (SLURM_JOB_ID % 9000) ))
else
    echo "[train-big] no SLURM_JOB_ID — cannot derive VLLM_PORT deterministically" >&2
    exit 1
fi
export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
echo "[train-big] waiting on $VLLM_URL/health ..."
attempts=0
until curl -sf $VLLM_URL/health > /dev/null; do
    sleep 5
    attempts=$((attempts + 1))
    if (( attempts > 240 )); then
        echo "[train-big] vLLM not ready after 20 min" >&2
        tail -200 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
done

# --- trainer with its own CUDA_VISIBLE_DEVICES ---------------------
export CUDA_VISIBLE_DEVICES="$TRAINER_VISIBLE"

if (( TRAINER_NUM_GPUS > 1 )); then
    LAUNCHER="accelerate launch --num_processes=$TRAINER_NUM_GPUS --num_machines=1 --mixed_precision=bf16"
else
    LAUNCHER="python"
fi

$LAUNCHER math_train.py \
    model_name="$MODEL_NAME" \
    lora_rank="$LORA_RANK" \
    group_size="$GROUP_SIZE" \
    groups_per_batch="$GROUPS_PER_BATCH" \
    learning_rate="$LEARNING_RATE" \
    env="$ENV" \
    dataset_schedule="$DATASET_SCHEDULE" \
    loss_fn="$LOSS_FN" \
    base_url="$VLLM_URL" \
    log_path="$LOG_DIR" \
    behavior_if_log_dir_exists=resume \
    ${=EXTRA_ARGS}
