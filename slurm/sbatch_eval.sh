#!/usr/bin/env zsh
#SBATCH --job-name=cllm-eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=babel-l9-32,babel-p9-20,babel-o5-16,babel-o5-32
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/home/ksnair/CuriousLLMs_logs/slurm-%x-%j.out
#
# Eval-only sbatch: launches vLLM, runs math_evaluation.py.

set -e

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/ksnair/worktrees/slurm}}"
source "$REPO_DIR/slurm/platform.sh"   # Babel vs Modal paths, envs, signals
cllm_activate clm

cllm_hf_cache shared
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LORA_RANK:=32}"
: "${VLLM_GPU_MEM_FRAC:=0.85}"   # eval-only: vLLM owns most of the GPU
: "${VLLM_TP_SIZE:=1}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    if [[ -n "${JOB_ID}" ]]; then
        # No timestamp: a requeued job (same job id) resumes into the same LOG_DIR.
        LOG_DIR=${LOGS_ROOT}/eval-${JOB_ID}-$(echo "$MODEL_NAME" | tr '/' '-')
    else
        STAMP=$(date +%Y%m%d-%H%M%S)
        LOG_DIR=${LOGS_ROOT}/eval-${STAMP}-${JOB_ID:-local}-$(echo "$MODEL_NAME" | tr '/' '-')
    fi
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[eval] LOG_DIR=$LOG_DIR"

cd "$REPO_DIR"
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
source slurm/launch_vllm.sh

EXTRA_MODEL_PATH=""
if [[ -n "${MODEL_PATH}" ]]; then
    EXTRA_MODEL_PATH="model_path=$MODEL_PATH"
fi

python math_evaluation.py \
    backend=local \
    model_name="$MODEL_NAME" \
    log_path="$LOG_DIR" \
    $EXTRA_MODEL_PATH \
    ${=EXTRA_ARGS}
