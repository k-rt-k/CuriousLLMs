# SLURM-native CuriousLLMs

This worktree replaces the Tinker (Thinking Machines) cloud API with a local
vLLM + PyTorch+PEFT stack so training and inference run entirely on Babel via
`sbatch`.

**For the full operator guide (pipeline overview, stage-by-stage recipes,
CLI reference, troubleshooting), see [../docs/pipeline.md](../docs/pipeline.md).**
This file covers SLURM-specific bits: env patches, memory rules, smoke tests.

## How it works

The adapter layer lives in `local_backend/`:

- `LocalServiceClient` — replaces `tinker.ServiceClient`.
- `LocalTrainingClient` — replaces `tinker.TrainingClient`. Holds the HF base
  model + a PEFT LoRA + AdamW. Implements the same async-future-shaped methods
  Tinker exposes (`forward_backward_async`, `optim_step_async`, `save_state_async`,
  `save_weights_for_sampler_async`, `load_state_async`, `create_sampling_client`).
- `LocalSamplingClient` — replaces `tinker.SamplingClient`. Async HTTP against
  vLLM's OpenAI-compatible `/v1/completions`. Returns genuine `tinker.SampleResponse`
  objects so `tinker_cookbook` consumers don't notice.
- `weight_sync.py` — after each train step the trainer writes a PEFT adapter
  directory and POSTs `/v1/load_lora_adapter` so vLLM serves the new policy.

The rest of `tinker_cookbook` (envs, renderers, RL types, rollouts, data
processing, KL/PPO metrics, evaluator base class, checkpoint_utils) is reused
unchanged via the git submodule.

## Manual env tweaks (gpt-oss / MoE LoRA)

Two changes were applied to the `clm` micromamba env to make `openai/gpt-oss-20b`
LoRA training work end-to-end on vLLM 0.20.2. **Both are env-local, not in this
repo.** If you recreate the env, re-apply both. A third, optional one (section 3) is
only needed to LoRA-adapt gpt-oss `lm_head`.

### 1. DeepGEMM (Hopper FP8 fast path for MoE)

vLLM 0.20.2 defaults to DeepGEMM for MoE FP8 attention paths (e.g. gpt-oss).
Without it, vLLM aborts with `RuntimeError: DeepGEMM backend is not available
or outdated`. Build from source (the PyPI package of the same name is
unrelated):

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$HOME/micromamba/envs/clm/bin:$PATH
mkdir -p ~/build_deepgemm && cd ~/build_deepgemm
curl -sO https://raw.githubusercontent.com/vllm-project/vllm/v0.20.2/tools/install_deepgemm.sh
bash install_deepgemm.sh
python -c "import deep_gemm; print(deep_gemm.__version__)"   # 2.5.0+891d57b
```

### 2. lora_shrink_op `.contiguous()` patch

vLLM 0.20.2 (and main) ship a triton kernel that asserts inputs are contiguous,
but the gpt-oss qkv_proj path feeds non-contiguous sliced views into it. We
patch the kernel to make the tensors contiguous instead of asserting.

```bash
F=$HOME/micromamba/envs/clm/lib/python3.11/site-packages/vllm/lora/ops/triton_ops/lora_shrink_op.py
cp "$F" "$F.orig"   # idempotent if .orig already exists
# In F, replace:
#     assert inputs.is_contiguous()
#     assert output_tensor.is_contiguous()
# with:
#     if not inputs.is_contiguous():
#         inputs = inputs.contiguous()
#     if not output_tensor.is_contiguous():
#         output_tensor = output_tensor.contiguous()
```

Revert via `cp "$F.orig" "$F"`. Upstream tracking: vLLM PR #40338 / bnellnm
suggested this `.contiguous()` bandaid; not merged as of vLLM main on 2026-05-12.

### 3. gpt-oss `lm_head` LoRA plugin (optional)

vLLM's `GptOssForCausalLM` declares no `embedding_modules`, so adapters containing
`lm_head` are rejected. `lean_rl/vllm_plugin` is a vLLM general plugin that sets
`embedding_modules = {"lm_head": "output_embeddings"}`. Install it into the env that
runs vLLM and enable it with `VLLM_PLUGINS` (vLLM loads plugins in every process):

```bash
pip install -e lean_rl/vllm_plugin
export VLLM_PLUGINS=lean_rl_gpt_oss_lm_head   # before `vllm serve`
```

Whether served logprobs match HF with expert/`lm_head` adapters is checked by
`python -m lean_rl.tools.spike_lora_serving` (see its docstring).

## Setup

```bash
# 1. From the worktree:
cd /home/ksnair/worktrees/slurm

# 2. Initialize tinker-cookbook (done once):
git submodule update --init --recursive

# 3. Install vLLM into your training env (fmg-style):
~/micromamba/envs/fmg/bin/pip install "vllm>=0.6.3,<0.8"

# 4. Make sure these are also present in the env (most already are in fmg):
~/micromamba/envs/fmg/bin/pip install chz==0.3.0 inspect_ai math-verify \
    latex2sympy2_extended google-genai blobfile universal_pathlib aioboto3 \
    s3fs jsonlines tenacity python-dotenv
```

## Submitting jobs

All scripts run on the preemptible partition (`--partition=preempt --qos=preempt_qos --requeue
--open-mode=append`). SLURM opens the `#SBATCH --output` file before the script runs, so create its
directory once before the first `sbatch`:

```bash
mkdir -p /home/ksnair/CuriousLLMs_logs
```

Scripts `cd` to `$REPO_DIR` (default: `$SLURM_SUBMIT_DIR`, else `/home/ksnair/worktrees/slurm`).
Under SLURM, the default `LOG_DIR` is derived from `SLURM_JOB_ID` only (no timestamp), so a requeued
job resumes into the same directory (`behavior_if_log_dir_exists=resume`). On preemption (SIGTERM)
`sbatch_train.sh` / `sbatch_train_bigmodel.sh` forward the signal to the trainer and vLLM and exit
non-zero.

### Small-model training (Llama-3B, Qwen-7B, etc., co-resident on one GPU)

```bash
cd /home/ksnair/worktrees/slurm

sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,LORA_RANK=32 \
       slurm/sbatch_train.sh
```

Optional env overrides (all have defaults):

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B` | HF model id |
| `LORA_RANK` | 32 | Also passed to vLLM `--max-lora-rank` |
| `GROUP_SIZE` | 16 | Rollouts per problem |
| `GROUPS_PER_BATCH` | 128 | Problems per batch |
| `LEARNING_RATE` | 7e-5 | |
| `ENV` | `mixed` | `arithmetic` / `math` / `mixed` / `polaris` / `deepmath` / `gsm8k` |
| `DATASET_SCHEDULE` | `m-m` | `e-h`, `m-h`, `e-m`, `m-m` |
| `LOSS_FN` | `ppo` | `ppo` or `importance_sampling` |
| `VLLM_GPU_MEM_FRAC` | 0.45 | vLLM's share of GPU; trainer takes the rest |
| `LOG_DIR` | auto-generated under `/data/hf_cache/ksnair/CuriousLLMs_logs/` | |
| `INIT_FROM_ADAPTER` | unset | Path to a LoRA adapter directory (e.g. a format SFT `states/format_final/`). Loads weights only — optimizer starts fresh, batch counter at 0. |
| `EXTRA_ARGS` | empty | Anything else, passed verbatim to `math_train.py` |

### Eval-only

```bash
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,MODEL_PATH=/path/to/adapter \
       slurm/sbatch_eval.sh
```

### Format SFT cold-start (pre-RL)

DeepSeek-R1-style: teach the LoRA the target output format (chat-template wrapping +
`\boxed{answer}`) on a HuggingFace dataset of solved problems, then warm-start RL
from the resulting adapter. Stops RL from having to learn formatting from random
init.

```bash
# Stage 1 — format cold-start
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,DATASET_NAME=AI-MO/NuminaMath-CoT,NUM_STEPS=500 \
       slurm/sbatch_format.sh
# -> produces /data/hf_cache/ksnair/CuriousLLMs_logs/<stamp>-<jobid>-format-...
#    The final adapter lives under that dir at  states/format_final/

# Stage 2 — RL warm-started from the format adapter
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,\
INIT_FROM_ADAPTER=/data/hf_cache/ksnair/CuriousLLMs_logs/<stage1>/states/format_final \
       slurm/sbatch_train.sh
```

`sbatch_format.sh` env vars (defaults in parentheses): `MODEL_NAME`
(`meta-llama/Llama-3.2-3B`), `LORA_RANK` (32), `DATASET_NAME`
(`AI-MO/NuminaMath-CoT`), `DATASET_SPLIT` (`train`), `DATASET_CONFIG` (unset;
e.g. `main` for `openai/gsm8k`), `PROBLEM_COLUMN` (`problem`),
`SOLUTION_COLUMN` (`solution`), `MAX_ROWS` (unset = all), `MAX_SEQ_LEN` (4096),
`BATCH_SIZE` (8), `NUM_STEPS` (500), `LEARNING_RATE` (1e-4), `SAVE_EVERY` (100),
`LOG_DIR` (auto).

The data side lives in `local_backend/format_dataset.py` — it loads the HF
dataset, applies the model's chat template, and builds response-masked Datums
(weights = 0 on prompt tokens, 1 on solution tokens) using
`tinker_cookbook.supervised.common.datum_from_tokens_weights`.

### Big-model (Llama-3.1-8B, Llama-3-8B)

```bash
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
       slurm/sbatch_train_bigmodel.sh
```

Multi-GPU trainer (single node): vLLM takes `VLLM_TP_SIZE` GPUs, the trainer takes the next
`TRAINER_NUM_GPUS` and runs under `torchrun --standalone --nproc_per_node=$TRAINER_NUM_GPUS`
(data-parallel LoRA; rank 0 drives, other ranks serve; requires `backend=local`):

```bash
sbatch --gres=gpu:A100_80GB:4 \
       --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-1B,VLLM_TP_SIZE=2,TRAINER_NUM_GPUS=2 \
       slurm/sbatch_train_bigmodel.sh
```

### Multi-node data-parallel training

```bash
sbatch --nodes=2 --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,VLLM_TP_SIZE=1 \
       slurm/sbatch_train_dist.sh
```

`sbatch_train_dist.sh` computes the topology and `srun`s `slurm/dist_node.sh` once per node
(`--ntasks-per-node=1 --kill-on-bad-exit=1`). Each node runs one vLLM on GPUs `0..VLLM_TP_SIZE-1` and
`TRAINER_GPUS_PER_NODE = GPUS_PER_NODE - VLLM_TP_SIZE` trainer ranks on the rest
(`LOCAL_TRAINER_GPU_MEM_FRAC=1.0`), launched with
`torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=$TRAINER_GPUS_PER_NODE
--rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT`. Each node waits for every node's vLLM
`/health` before starting torchrun. Global rank 0 (any node: c10d assigns ranks in join order) drives the loop; its sampling client balances across
`VLLM_URLS`.

| Var | Default | Notes |
| --- | --- | --- |
| `--nodes` | 2 | Node count (`NNODES` = `$SLURM_NNODES`) |
| `GPUS_PER_NODE` | `$SLURM_GPUS_ON_NODE`, else 4 | Header requests `--gres=gpu:4`, 16 CPUs per node |
| `VLLM_TP_SIZE` | 1 | GPUs per node given to that node's vLLM |
| `MASTER_ADDR` | first host of `scontrol show hostnames` | Short hostnames (resolve inside Babel) |
| `MASTER_PORT` | `29500 + JOB_ID % 1000` | torchrun rendezvous |
| `VLLM_PORT` | `20000 + JOB_ID % 9000` | `VLLM_URLS` = every host at this port |
| `DRY_RUN` | 0 | `1` prints topology and the srun/torchrun command lines, launches nothing |

All other knobs match `sbatch_train.sh` (`MODEL_NAME`, `LORA_RANK`, `INIT_FROM_ADAPTER`, `EXTRA_ARGS`, ...);
`LOG_DIR` defaults to `/data/hf_cache/ksnair/CuriousLLMs_logs/${SLURM_JOB_ID}-<model>` (shared storage, so a
requeue resumes into it). On SIGTERM the signal is forwarded to the `srun` step, then to each node's torchrun and
vLLM. Per-node vLLM logs are in `$LOG_DIR/node<N>/vllm.log`. Not yet verified on a real 2-node job: that Babel
accepts the `srun` step (fallback: `ssh` into allocated nodes), and requeue-resume.

New config flags (via `EXTRA_ARGS`): `backend` (`tinker`|`local`; scripts pass `local`), `kl_reference`
(`base`|`init`; `init` uses the `init_from_adapter` warm-start adapter as the KL reference), `entropy_bonus_coef`
(float, unset = disabled).

`math_evaluation.py` stays single-process (it errors if `WORLD_SIZE>1`) and reads `$VLLM_URLS`
(comma-separated), then `$VLLM_URL`, for the local backend.

## Memory rules of thumb

| GPU | vLLM share | Trainer share | Comfortable model size |
| --- | --- | --- | --- |
| L40S 48 GB | 0.45 | 0.45 | ≤ 3B with LoRA |
| A100 80 GB | 0.45 | 0.45 | ≤ 8B with LoRA |
| H100 80 GB | 0.45 | 0.45 | ≤ 8B with LoRA |

For models that won't co-reside (e.g. gpt-oss-20b on a single 80 GB GPU), the
plan is a **time-multiplex mode** that pages the HF model to CPU during
sampling and uses vLLM's `/sleep` + `/wake_up` endpoints during training. This
is not implemented yet — see the plan file at
`~/.claude/plans/make-a-new-worktree-humming-gizmo.md`.

## Resume

`math_train.py` writes `checkpoints.jsonl` under `$LOG_DIR`. Re-running with the
same `log_path` will pick up the latest `state_path` automatically (the existing
resume logic in `curiosity_train.py` is preserved).

## Smoke tests (after vLLM is installed)

```bash
# 1. Quick sanity: launch vLLM standalone
LOG_DIR=/tmp/vllm-smoke MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
    source slurm/launch_vllm.sh
curl -s $VLLM_URL/v1/models | python -m json.tool

# 2. End-to-end tiny training run (request a debug GPU first):
srun --partition=debug --gres=gpu:L40S:1 --time=1:00:00 --pty zsh
# inside the interactive shell:
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct LORA_RANK=8 GROUP_SIZE=4 \
GROUPS_PER_BATCH=4 EXTRA_ARGS="env=arithmetic eval_every=2 save_every=2 max_tokens=128" \
    bash slurm/sbatch_train.sh
```

## Disk usage

Each training step writes a fresh adapter directory under
`$LOG_DIR/adapters/step_NNNNNN/` (~30 MB). For long runs, run periodically:

```bash
find $LOG_DIR/adapters -maxdepth 1 -type d -name 'step_*' | sort | head -n -10 | xargs -r rm -rf
```

(A real janitor inside `LocalTrainingClient` is a TODO.)
