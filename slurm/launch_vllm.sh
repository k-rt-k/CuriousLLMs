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
#   VLLM_URL              http://127.0.0.1:$VLLM_PORT
#   VLLM_PID              child process id

: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${LOG_DIR:?LOG_DIR must be set}"
: "${LORA_RANK:=32}"
: "${VLLM_GPU_MEM_FRAC:=0.45}"
: "${VLLM_DTYPE:=bfloat16}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_TP_SIZE:=1}"
: "${VLLM_EXTRA_ARGS:=}"

if [[ -z "${VLLM_PORT}" ]]; then
    # Derive a deterministic port from SLURM_JOB_ID so concurrent jobs on the
    # same node never collide. Fall back to a random port outside any job.
    if [[ -n "${SLURM_JOB_ID}" ]]; then
        VLLM_PORT=$(( 20000 + (SLURM_JOB_ID % 9000) ))
    else
        VLLM_PORT=$(shuf -i 20000-28999 -n 1)
    fi
fi
export VLLM_PORT
export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"

mkdir -p "$LOG_DIR"

echo "[launch_vllm] starting vLLM model=$MODEL_NAME port=$VLLM_PORT gpu_mem=$VLLM_GPU_MEM_FRAC tp=$VLLM_TP_SIZE"
# Required for /v1/{load,unload}_lora_adapter to be served (vLLM 0.15+).
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
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

echo "[launch_vllm] waiting for vLLM at $VLLM_URL/health ..."
attempts=0
until curl -sf "$VLLM_URL/health" > /dev/null; do
    sleep 3
    attempts=$((attempts + 1))
    if (( attempts > 200 )); then
        echo "[launch_vllm] vLLM did not come up in time" >&2
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
