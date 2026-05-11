# SLURM-native CuriousLLMs

This worktree replaces the Tinker (Thinking Machines) cloud API with a local
vLLM + PyTorch+PEFT stack so training and inference run entirely on Babel via
`sbatch`.

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
| `LOG_DIR` | auto-generated under `/data/user_data/ksnair/CuriousLLMs_logs/` | |
| `EXTRA_ARGS` | empty | Anything else, passed verbatim to `math_train.py` |

### Eval-only

```bash
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.2-3B,MODEL_PATH=/path/to/adapter \
       slurm/sbatch_eval.sh
```

### Big-model (Llama-3.1-8B, Llama-3-8B)

```bash
sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
       slurm/sbatch_train_bigmodel.sh
```

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
