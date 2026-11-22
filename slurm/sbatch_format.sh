#!/usr/bin/env zsh
#SBATCH --job-name=cllm-format
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=babel-l9-32,babel-p9-20,babel-o5-16,babel-o5-32
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/home/ksnair/CuriousLLMs_logs/slurm-%x-%j.out
#
# Format SFT (cold-start) stage — DeepSeek-R1 style. Teaches the LoRA the
# target output format (chat-template + \boxed{answer}) before RL kicks in.
# Output adapter is then warm-started by sbatch_train.sh via INIT_FROM_ADAPTER.
#
# Override defaults via the environment when sbatch'ing:
#   sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,\
#                   DATASET_NAME=AI-MO/NuminaMath-CoT,NUM_STEPS=500 \
#          slurm/sbatch_format.sh
#
# Optional env vars (all have defaults):
#   MODEL_NAME       (default meta-llama/Llama-3.2-3B)
#   LORA_RANK        (default 32)
#   DATASET_NAME     (default AI-MO/NuminaMath-CoT)
#   DATASET_SPLIT    (default train)
#   DATASET_CONFIG   (default unset; e.g. "main" for gsm8k)
#   PROBLEM_COLUMN   (default problem)
#   SOLUTION_COLUMN  (default solution)
#   MAX_ROWS         (default unset = use all)
#   MAX_SEQ_LEN      (default 4096)
#   BATCH_SIZE       (default 8)
#   NUM_STEPS        (default 500)
#   LEARNING_RATE    (default 1e-4)
#   SAVE_EVERY       (default 100)
#   LOG_DIR          (default auto-generated under ${LOGS_ROOT}/)
#   EXTRA_ARGS       (passed verbatim to format_train.py)

set -e

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/ksnair/worktrees/slurm}}"
source "$REPO_DIR/slurm/platform.sh"   # Babel vs Modal paths, envs, signals
cllm_activate clm

# /data/hf_cache/{hub,datasets} is not writable from compute nodes for our user (HF file
# locks need write access): per-user caches on Babel (slurm/platform.sh).
cllm_hf_cache user
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- defaults ------------------------------------------------------
: "${MODEL_NAME:=meta-llama/Llama-3.2-3B}"
: "${LORA_RANK:=32}"
: "${DATASET_NAME:=AI-MO/NuminaMath-CoT}"
: "${DATASET_SPLIT:=train}"
: "${PROBLEM_COLUMN:=problem}"
: "${SOLUTION_COLUMN:=solution}"
: "${MAX_SEQ_LEN:=4096}"
: "${BATCH_SIZE:=8}"
: "${NUM_STEPS:=500}"
: "${LEARNING_RATE:=1e-4}"
: "${SAVE_EVERY:=100}"
: "${VLLM_GPU_MEM_FRAC:=0.45}"
: "${VLLM_TP_SIZE:=1}"
: "${EXTRA_ARGS:=}"

if [[ -z "${LOG_DIR}" ]]; then
    if [[ -n "${JOB_ID}" ]]; then
        # No timestamp: a requeued job (same job id) resumes into the same LOG_DIR.
        LOG_DIR=${LOGS_ROOT}/${JOB_ID}-format-$(echo "$MODEL_NAME" | tr '/' '-')
    else
        STAMP=$(date +%Y%m%d-%H%M%S)
        LOG_DIR=${LOGS_ROOT}/${STAMP}-${JOB_ID:-local}-format-$(echo "$MODEL_NAME" | tr '/' '-')
    fi
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[format] LOG_DIR=$LOG_DIR"

# --- vLLM ----------------------------------------------------------
# We still launch vLLM for parity with sbatch_train.sh — it's idle during SFT
# but lets format_final/ get hot-loaded for a final test sample if desired.
cd "$REPO_DIR"
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
# Node-local Triton cache for vLLM AND the trainer (~/.triton may point at a
# /scratch dir missing on this node). Set here in the parent scope because
# launch_vllm.sh (same default) may run in a subshell whose exports are lost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER}-${JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"
source slurm/launch_vllm.sh

# --- training ------------------------------------------------------
python format_train.py \
    model_name="$MODEL_NAME" \
    lora_rank="$LORA_RANK" \
    dataset_name="$DATASET_NAME" \
    dataset_split="$DATASET_SPLIT" \
    problem_column="$PROBLEM_COLUMN" \
    solution_column="$SOLUTION_COLUMN" \
    max_seq_len="$MAX_SEQ_LEN" \
    batch_size="$BATCH_SIZE" \
    num_steps="$NUM_STEPS" \
    learning_rate="$LEARNING_RATE" \
    save_every="$SAVE_EVERY" \
    base_url="$VLLM_URL" \
    log_path="$LOG_DIR" \
    behavior_if_log_dir_exists=resume \
    ${DATASET_CONFIG:+dataset_config="$DATASET_CONFIG"} \
    ${MAX_ROWS:+max_rows="$MAX_ROWS"} \
    ${=EXTRA_ARGS}
