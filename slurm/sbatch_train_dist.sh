#!/usr/bin/env zsh
#SBATCH --job-name=cllm-train-dist
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=babel-l9-32,babel-p9-20,babel-o5-16,babel-o5-32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kartiknsree@gmail.com
#SBATCH --output=/home/ksnair/CuriousLLMs_logs/slurm-%x-%j.out
#
# Multi-node data-parallel RL training (local backend).
#
# Every node runs ONE vLLM server on GPUs 0..VLLM_TP_SIZE-1 and
# TRAINER_GPUS_PER_NODE = GPUS_PER_NODE - VLLM_TP_SIZE torchrun trainer ranks on
# the remaining GPUs (see slurm/dist_node.sh). Global rank 0 (on whichever node c10d
# rendezvous assigns it -- not necessarily the first) drives the RL loop; its sampling client load-balances across every node's vLLM
# (VLLM_URLS). Only rank 0 logs, checkpoints and talks to vLLM.
#
# Node count: sbatch --nodes=N (default 2). GPUs per node: GPUS_PER_NODE
# (default $SLURM_GPUS_ON_NODE, else 4). VLLM_TP_SIZE defaults to 1.
#
# Override defaults via the environment when sbatch'ing:
#   sbatch --nodes=2 --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,VLLM_TP_SIZE=1 \
#          slurm/sbatch_train_dist.sh
#
# Preemption: SIGTERM is forwarded to the srun step (which signals every node's
# dist_node.sh -> torchrun + vLLM). The requeued job reuses the same LOG_DIR
# and resumes (behavior_if_log_dir_exists=resume).
#
# Modal (modal_jobs/app.py, CLLM_PLATFORM=modal): every node is one container that runs this
# script; it execs slurm/dist_node.sh directly (no srun) with node index CLLM_NODE_RANK.
#
# DRY_RUN=1 prints the computed topology and the srun/torchrun command lines
# without launching anything (used for CPU-side checks).

set -e

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/ksnair/worktrees/slurm}}"
cd "$REPO_DIR"
source "$REPO_DIR/slurm/platform.sh"   # Babel vs Modal paths, envs, hosts, signals

: "${MODEL_NAME:=meta-llama/Llama-3.2-3B}"
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
: "${EXTRA_ARGS:=}"
: "${DRY_RUN:=0}"

: "${JOB_NUM:?no job id: run under sbatch (SLURM_JOB_ID) or via modal_jobs/app.py (JOB_ID)}"

TRAINER_GPUS_PER_NODE=$(( GPUS_PER_NODE - VLLM_TP_SIZE ))
if (( TRAINER_GPUS_PER_NODE < 1 )); then
    echo "[train-dist] no trainer GPUs: GPUS_PER_NODE=$GPUS_PER_NODE VLLM_TP_SIZE=$VLLM_TP_SIZE" >&2
    exit 1
fi

# Babel: short hostnames (they resolve inside Babel); Modal: container IPs. The same names serve
# MASTER_ADDR and VLLM_URLS.
HOSTS=(${(f)"$(cllm_hosts)"})
NNODES=${#HOSTS[@]}
MASTER_ADDR=${HOSTS[1]}
MASTER_PORT=$(( 29500 + JOB_NUM % 1000 ))
VLLM_PORT=$(( 20000 + JOB_NUM % 9000 ))
URLS=(); for h in "${HOSTS[@]}"; do URLS+=("http://$(cllm_url_host "$h"):${VLLM_PORT}"); done
VLLM_URLS=${(j:,:)URLS}

if [[ -z "${LOG_DIR}" ]]; then
    # No timestamp: a requeued job (same job id) resumes into the same LOG_DIR.
    LOG_DIR=${LOGS_ROOT}/${JOB_ID}-$(echo "$MODEL_NAME" | tr '/' '-')
fi

export MODEL_NAME LORA_RANK GROUP_SIZE GROUPS_PER_BATCH LEARNING_RATE ENV DATASET_SCHEDULE LOSS_FN
export VLLM_GPU_MEM_FRAC VLLM_TP_SIZE GPUS_PER_NODE TRAINER_GPUS_PER_NODE EXTRA_ARGS DRY_RUN
export MASTER_ADDR MASTER_PORT VLLM_PORT VLLM_URLS LOG_DIR REPO_DIR
[[ -n "${INIT_FROM_ADAPTER}" ]] && export INIT_FROM_ADAPTER

echo "[train-dist] nodes=$NNODES hosts=${HOSTS[*]}"
echo "[train-dist] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT VLLM_PORT=$VLLM_PORT"
echo "[train-dist] VLLM_URLS=$VLLM_URLS"
echo "[train-dist] per node: GPUs=$GPUS_PER_NODE vLLM_tp=$VLLM_TP_SIZE trainer_ranks=$TRAINER_GPUS_PER_NODE"
echo "[train-dist] LOG_DIR=$LOG_DIR"

if [[ "$CLLM_PLATFORM" == modal ]]; then
    exec zsh "$REPO_DIR/slurm/dist_node.sh"   # this container is node $CLLM_NODE_RANK
fi

SRUN_CMD=(srun --ntasks-per-node=1 --kill-on-bad-exit=1 zsh "$REPO_DIR/slurm/dist_node.sh")
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[train-dist] DRY_RUN srun: ${SRUN_CMD[*]}"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Background + wait so the TERM trap can forward preemption to the srun step.
on_term() {
    echo "[train-dist] SIGTERM (preempted?) - forwarding to srun step" >&2
    set +e  # errexit must not abort the handler (wait returns 143 after the kill)
    [[ -n "$SRUN_PID" ]] && kill -TERM "$SRUN_PID" 2>/dev/null
    [[ -n "$SRUN_PID" ]] && wait "$SRUN_PID" 2>/dev/null
    exit 143
}
cllm_trap_preempt on_term

"${SRUN_CMD[@]}" &
SRUN_PID=$!
wait $SRUN_PID
