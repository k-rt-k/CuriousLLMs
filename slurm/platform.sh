#!/usr/bin/env zsh
# Sourced by every slurm/*.sh (after REPO_DIR is known): everything that differs between Babel (sbatch) and Modal
# (modal_jobs/app.py, which runs these same scripts inside a container).
#
#   CLLM_PLATFORM   babel (default) | modal
#
# Exports / defines:
#   JOB_ID          run id: $SLURM_JOB_ID on Babel (may be empty outside sbatch); on Modal the run
#                   id from modal_jobs/app.py, stable across Modal's preemption retries
#   JOB_NUM         numeric form of JOB_ID, for port derivation (empty if JOB_ID is)
#   LOGS_ROOT       parent of the default LOG_DIRs
#   CLLM_USER_CACHE per-user writable cache root (tiktoken vocab etc.)
#   PREEMPT_WAIT_S  how long a script waits for its ranks to save after a preemption signal
#   cllm_activate ENV        activate a Python env (micromamba on Babel, a venv on Modal)
#   cllm_env_python ENV      that env's python
#   cllm_hf_cache shared|user  set HF_HUB_CACHE / HF_DATASETS_CACHE
#   cllm_hosts               the job's node hostnames/IPs, one per line, node 0 first
#   cllm_gpus_on_node        GPUs allocated on this node
#   cllm_self_host           an address other processes use to reach this node
#   cllm_url_host H          H, bracketed if it is an IPv6 literal (for URLs)
#   cllm_trap_preempt FN     run FN on the platform's preemption signal(s)
#
# On Modal (from modal_jobs/app.py): JOB_ID, CLLM_HOSTS (comma list, node 0 first),
# CLLM_NODE_RANK, CLLM_SELF_HOST.

: "${CLLM_PLATFORM:=babel}"

case "$CLLM_PLATFORM" in
babel)
    : "${JOB_ID:=${SLURM_JOB_ID:-}}"
    : "${LOGS_ROOT:=/data/hf_cache/ksnair/CuriousLLMs_logs}"
    : "${CLLM_USER_CACHE:=/data/hf_cache/${USER}}"
    : "${PREEMPT_WAIT_S:=100}"   # Babel: GraceTime 0, KillWait 120 s; leave room to stop vLLM
    export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
    export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
    ;;
modal)
    : "${JOB_ID:?JOB_ID must be set (modal_jobs/app.py sets it)}"
    : "${LOGS_ROOT:=/vol/logs}"
    : "${CLLM_USER_CACHE:=/vol/cache}"
    : "${PREEMPT_WAIT_S:=25}"    # Modal: 30 s from the interrupt signal to the kill
    : "${CLLM_HOSTS:=127.0.0.1}"
    : "${CLLM_NODE_RANK:=0}"
    # Modal's preemption notice is an interrupt: the lean trainer treats SIGINT like SIGTERM.
    export LEAN_PREEMPT_ON_SIGINT=1
    ;;
*)
    echo "[platform] unknown CLLM_PLATFORM=$CLLM_PLATFORM (babel|modal)" >&2
    exit 1
    ;;
esac
export CLLM_PLATFORM JOB_ID LOGS_ROOT CLLM_USER_CACHE PREEMPT_WAIT_S

if [[ -z "$JOB_ID" ]]; then
    JOB_NUM=
elif [[ "$JOB_ID" == <-> ]]; then
    JOB_NUM=$JOB_ID
else
    JOB_NUM=$(printf '%s' "$JOB_ID" | cksum | cut -d' ' -f1)
fi
export JOB_NUM

cllm_activate() {
    local env=$1
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        source "/opt/envs/$env/bin/activate"
        export CLLM_ENV_PREFIX="$VIRTUAL_ENV"
    else
        if ! typeset -f micromamba > /dev/null; then
            eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
        fi
        micromamba activate "$env"
        export CLLM_ENV_PREFIX="$CONDA_PREFIX"
    fi
}

cllm_env_python() {
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        echo "/opt/envs/$1/bin/python"
    else
        echo "$MAMBA_ROOT_PREFIX/envs/$1/bin/python"
    fi
}

# shared: Babel's community cache (read-mostly, works for the RL flow); user: per-user writable
# cache (format SFT needs write access for the datasets/hub file locks). One cache on Modal.
cllm_hf_cache() {
    unset HF_HOME
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        export HF_HUB_CACHE=/vol/hf/hub HF_DATASETS_CACHE=/vol/hf/datasets
    elif [[ "$1" == user ]]; then
        export HF_HUB_CACHE=/data/hf_cache/ksnair/.hf_hub_cache
        export HF_DATASETS_CACHE=/data/hf_cache/ksnair/.hf_datasets_cache
    else
        export HF_HUB_CACHE=/data/hf_cache/hub HF_DATASETS_CACHE=/data/hf_cache/datasets
    fi
    [[ "$CLLM_PLATFORM" == modal || "$1" == user ]] && mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
    return 0
}

cllm_hosts() {
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        print -l ${(s:,:)CLLM_HOSTS}
    else
        : "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST must be set (run under sbatch)}"
        scontrol show hostnames "$SLURM_JOB_NODELIST"
    fi
}

cllm_gpus_on_node() {
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    else
        echo "${SLURM_GPUS_ON_NODE:-}"
    fi
}

cllm_self_host() {
    if [[ "$CLLM_PLATFORM" == modal ]]; then
        echo "${CLLM_SELF_HOST:-127.0.0.1}"
    else
        hostname -f
    fi
}

cllm_url_host() {
    if [[ "$1" == *:* ]]; then echo "[$1]"; else echo "$1"; fi
}

# Babel preempts with SIGTERM. Modal sends an interrupt, possibly to every process: an INT must
# then also mean "let the ranks save", not "stop vLLM now".
cllm_trap_preempt() {
    trap "$1" TERM
    [[ "$CLLM_PLATFORM" == modal ]] && trap "$1" INT
    return 0
}
