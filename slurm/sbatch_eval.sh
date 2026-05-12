#!/usr/bin/env zsh
#SBATCH --job-name=cllm-eval
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/data/hf_cache/ksnair/CuriousLLMs_logs/slurm-eval-%j.out
#
# Eval-only sbatch: launches vLLM, runs math_evaluation.py.

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

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LORA_RANK:=32}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"   # eval-only: vLLM owns most of the GPU
: "${VLLM_TP_SIZE:=1}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR=/data/hf_cache/ksnair/CuriousLLMs_logs/eval-${STAMP}-${SLURM_JOB_ID:-local}-$(echo "$MODEL_NAME" | tr '/' '-')
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[eval] LOG_DIR=$LOG_DIR"

cd /home/ksnair/worktrees/slurm
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
source slurm/launch_vllm.sh

EXTRA_MODEL_PATH=""
if [[ -n "${MODEL_PATH}" ]]; then
    EXTRA_MODEL_PATH="--model_path $MODEL_PATH"
fi

python math_evaluation.py \
    --model_name "$MODEL_NAME" \
    --log_path "$LOG_DIR" \
    $EXTRA_MODEL_PATH \
    ${=EXTRA_ARGS}
