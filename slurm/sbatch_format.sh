#!/usr/bin/env zsh
#SBATCH --job-name=cllm-format
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/data/hf_cache/ksnair/CuriousLLMs_logs/slurm-%j.out
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
#   LOG_DIR          (default auto-generated under /data/hf_cache/ksnair/CuriousLLMs_logs/)
#   EXTRA_ARGS       (passed verbatim to format_train.py)

set -e

# --- micromamba ----------------------------------------------------
export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate clm

# --- HF cache (shared, writable for hpcuser) -----------------------
# NOTE: /data/hf_cache/{hub,datasets} is the community-shared cache and is
# NOT writable from compute nodes for our user — both HF datasets and HF Hub
# acquire filelock-style locks on read which require write access. Force the
# writable per-user paths here regardless of the inherited shell env (the
# user's ~/.zshrc may set HF_DATASETS_CACHE to the shared path).
unset HF_HOME
export HF_HUB_CACHE=/data/hf_cache/ksnair/.hf_hub_cache
export HF_DATASETS_CACHE=/data/hf_cache/ksnair/.hf_datasets_cache
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
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
    STAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR=/data/hf_cache/ksnair/CuriousLLMs_logs/${STAMP}-${SLURM_JOB_ID:-local}-format-$(echo "$MODEL_NAME" | tr '/' '-')
fi
export LOG_DIR
mkdir -p "$LOG_DIR"
echo "[format] LOG_DIR=$LOG_DIR"

# --- vLLM ----------------------------------------------------------
# We still launch vLLM for parity with sbatch_train.sh — it's idle during SFT
# but lets format_final/ get hot-loaded for a final test sample if desired.
cd /home/ksnair/worktrees/slurm
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
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
