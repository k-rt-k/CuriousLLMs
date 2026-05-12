#!/usr/bin/env zsh
#SBATCH --job-name=cllm-train
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/data/hf_cache/ksnair/CuriousLLMs_logs/slurm-%j.out
#
# Small-model RL training on Babel (single GPU, co-resident vLLM + trainer).
#
# Override defaults via the environment when sbatch'ing:
#   sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,LORA_RANK=32 \
#          slurm/sbatch_train.sh
#
# Optional env vars (all have defaults):
#   MODEL_NAME       (default meta-llama/Llama-3.2-3B)
#   LORA_RANK        (default 32)
#   GROUP_SIZE       (default 16)
#   GROUPS_PER_BATCH (default 128)
#   LEARNING_RATE    (default 7e-5)
#   ENV              (default mixed)
#   DATASET_SCHEDULE (default m-m)
#   LOSS_FN          (default ppo)
#   LOG_DIR          (default auto-generated under /data/hf_cache/ksnair/CuriousLLMs_logs/)
#   EXTRA_ARGS       (passed verbatim to math_train.py)

set -e

# --- micromamba ----------------------------------------------------
# clm env has vllm 0.15.1 + torch + peft + transformers already installed.
export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate clm

# --- HF cache (shared, writable for hpcuser) -----------------------
# Avoid /data/user_data/ksnair entirely — use the shared /data/hf_cache.
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
unset HF_HOME  # let HF default; HF_HUB_CACHE is the load-bearing one
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- defaults ------------------------------------------------------
: "${MODEL_NAME:=meta-llama/Llama-3.2-3B}"
: "${LORA_RANK:=32}"
: "${GROUP_SIZE:=16}"
: "${GROUPS_PER_BATCH:=128}"
: "${LEARNING_RATE:=7e-5}"
: "${ENV:=mixed}"
: "${DATASET_SCHEDULE:=m-m}"
: "${LOSS_FN:=ppo}"
: "${VLLM_GPU_MEM_FRAC:=0.45}"
: "${VLLM_TP_SIZE:=1}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR=/data/hf_cache/ksnair/CuriousLLMs_logs/${STAMP}-${SLURM_JOB_ID:-local}-$(echo "$MODEL_NAME" | tr '/' '-')
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[train] LOG_DIR=$LOG_DIR"

# --- vLLM ----------------------------------------------------------
cd /home/ksnair/worktrees/slurm
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
source slurm/launch_vllm.sh

# --- training ------------------------------------------------------
python math_train.py \
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
