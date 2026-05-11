#!/usr/bin/env zsh
#SBATCH --job-name=cllm-train-big
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/data/user_data/ksnair/CuriousLLMs_logs/slurm-%j.out
#
# Big-model RL training (e.g. gpt-oss-20b).
#
# IMPORTANT: time-multiplex mode (trainer/sampler swap GPU ownership) is not
# yet implemented in the local backend. For now this script runs the same
# co-resident pattern as sbatch_train.sh but on an 80 GB GPU, which is enough
# for ~7-8B models with vLLM + PEFT side-by-side. For 20B MoE (gpt-oss-20b)
# this will OOM — see slurm/README.md for the deferred big-model plan.

set -e

export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate fmg

export HF_HOME=/data/user_data/ksnair/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

: "${MODEL_NAME:=meta-llama/Llama-3.1-8B-Instruct}"
: "${LORA_RANK:=32}"
: "${GROUP_SIZE:=16}"
: "${GROUPS_PER_BATCH:=64}"
: "${LEARNING_RATE:=5e-6}"
: "${ENV:=mixed}"
: "${DATASET_SCHEDULE:=m-m}"
: "${LOSS_FN:=ppo}"
: "${VLLM_GPU_MEM_FRAC:=0.55}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR=/data/user_data/ksnair/CuriousLLMs_logs/${STAMP}-${SLURM_JOB_ID:-local}-$(echo "$MODEL_NAME" | tr '/' '-')
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[train-big] LOG_DIR=$LOG_DIR"

cd /home/ksnair/worktrees/slurm
source slurm/launch_vllm.sh

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
    ${=EXTRA_ARGS}
