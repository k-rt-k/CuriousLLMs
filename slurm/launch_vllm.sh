#!/usr/bin/env zsh
# Sourced helper. Launches vLLM in the background, exports VLLM_URL,
# and sets a trap so the caller's exit cleans it up.
#
# Inputs (env):
#   MODEL_NAME            (required)
#   LORA_RANK             (default 32)
#   VLLM_PORT             (default: random in 20000-29999)
#   VLLM_GPU_MEM_FRAC     (default 0.45)  -- co-resident with trainer
#   VLLM_DTYPE            (default bfloat16)
#   VLLM_MAX_MODEL_LEN    (default 4096)
#   VLLM_EXTRA_ARGS       (optional extra flags)
#   LOG_DIR               (required; where vllm.log lives)
#
# Outputs (env):
#   VLLM_URL              http://<this node>:$VLLM_PORT (cllm_self_host; server binds 0.0.0.0)
#   VLLM_PID              child process id

: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${LOG_DIR:?LOG_DIR must be set}"
: "${LORA_RANK:=32}"
: "${VLLM_GPU_MEM_FRAC:=0.45}"
: "${VLLM_DTYPE:=bfloat16}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_TP_SIZE:=1}"
# ~/.triton may be a symlink into node-local /scratch that does not exist on
# every node (vLLM engine init then dies with FileNotFoundError). Use a
# per-job node-local dir instead. The calling sbatch scripts / dist_node.sh
# export the same default in their own scope (this file may be sourced in a
# subshell), so the trainer gets it too; an existing value is kept.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton-${USER}-${JOB_ID:-${SLURM_JOB_ID:-local}}}"
mkdir -p "$TRITON_CACHE_DIR"
# openai_harmony (gpt-oss tokenizer in vLLM) caches vocab files in /tmp/tiktoken-rs-cache by
# default; on shared nodes another user may own it (Permission denied -> vLLM dies at startup).
# Persistent per-user dir (a per-job $TMPDIR would re-download the vocab every job).
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_DIR:-${CLLM_USER_CACHE:-/data/hf_cache/${USER}}/tiktoken-rs}"
mkdir -p "$TIKTOKEN_RS_CACHE_DIR"
: "${VLLM_EXTRA_ARGS:=}"

if [[ -z "${VLLM_PORT}" ]]; then
    # Derive a deterministic port from the job id (JOB_NUM, slurm/platform.sh) so concurrent
    # jobs on the same node never collide. Fall back to a random port outside any job.
    : "${JOB_NUM:=${SLURM_JOB_ID:-}}"
    if [[ -n "${JOB_NUM}" ]]; then
        VLLM_PORT=$(( 20000 + (JOB_NUM % 9000) ))
    else
        VLLM_PORT=$(shuf -i 20000-28999 -n 1)
    fi
fi
export VLLM_PORT
if typeset -f cllm_self_host > /dev/null; then
    export VLLM_URL="http://$(cllm_url_host "$(cllm_self_host)"):${VLLM_PORT}"
else
    export VLLM_URL="http://$(hostname -f):${VLLM_PORT}"
fi

mkdir -p "$LOG_DIR"

echo "[launch_vllm] starting vLLM model=$MODEL_NAME port=$VLLM_PORT gpu_mem=$VLLM_GPU_MEM_FRAC tp=$VLLM_TP_SIZE"
# Required for /v1/{load,unload}_lora_adapter to be served (vLLM 0.15+).
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# On vLLM 0.20.2 (`clm` env) gpt-oss needs DeepGEMM, installed in `clm` — see
# slurm/README.md "Manual env tweaks". The default server env `clm-vllm` (vLLM 0.33)
# needs no patches. To force the Marlin fallback on 0.20.2, set VLLM_USE_DEEP_GEMM=0
# and VLLM_MOE_USE_DEEP_GEMM=0 before sbatch.
# --generation-config vllm: ignore the model's generation_config.json sampling defaults
# (Qwen3 ships temperature 0.6 / top_k 20 / top_p 0.95); RL sampling must use exactly the
# request's parameters, as Tinker does.
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank "$LORA_RANK" \
    --max-cpu-loras 8 \
    --gpu-memory-utilization "$VLLM_GPU_MEM_FRAC" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --enforce-eager \
    --generation-config vllm \
    ${=VLLM_EXTRA_ARGS} \
    > "$LOG_DIR/vllm.log" 2>&1 &
export VLLM_PID=$!

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[launch_vllm] terminating vLLM pid=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup_vllm EXIT INT TERM

# Cold reads from /data are slow on Babel (Qwen3-8B weights took 801 s), so the default is 40 min.
: "${VLLM_STARTUP_TIMEOUT_S:=7200}"
echo "[launch_vllm] waiting for vLLM at $VLLM_URL/health (timeout ${VLLM_STARTUP_TIMEOUT_S}s) ..."
attempts=0
until curl -sf "$VLLM_URL/health" > /dev/null; do
    sleep 3
    attempts=$((attempts + 1))
    if (( attempts * 3 > VLLM_STARTUP_TIMEOUT_S )); then
        echo "[launch_vllm] vLLM did not come up in ${VLLM_STARTUP_TIMEOUT_S}s" >&2
        tail -n 100 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[launch_vllm] vLLM exited prematurely" >&2
        tail -n 100 "$LOG_DIR/vllm.log" >&2 || true
        exit 1
    fi
done
echo "[launch_vllm] vLLM ready at $VLLM_URL"
