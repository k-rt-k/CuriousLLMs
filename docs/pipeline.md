# CuriousLLMs SLURM Pipeline — Operator Guide

End-to-end docs for the `slurm` branch: what runs locally on Babel via SLURM, what's ready, what's not, and how to chain the stages.

The full pipeline:

```
              ┌────────────────────┐      ┌──────────────────────┐
HF dataset ─▶ │  format SFT        │ ───▶ │  RL (curiosity+PPO)  │ ───▶ AIME / eval
              │  format_train.py   │      │  math_train.py       │
              └────────────────────┘      └──────────────────────┘
                       ▲                            ▲
                       │                            │
              slurm/sbatch_format.sh    slurm/sbatch_train.sh
                                        slurm/sbatch_train_bigmodel.sh
```

Both training stages share `local_backend/` (vLLM HTTP client + PyTorch+PEFT LoRA trainer with an AdamW optimizer). The output adapter of stage 1 is the warm-start input to stage 2.

---

## Status

### Ready

- **vLLM-backed sampling** with hot-swap LoRA reloads via `/v1/load_lora_adapter` (`local_backend/sampling_client.py`, `weight_sync.py`).
- **LoRA training loop** on a single (or multi-)GPU node — HF + PEFT + AdamW, with PPO and importance-sampling losses (`local_backend/training_client.py`, `losses.py`).
- **Format SFT cold-start** (DeepSeek-R1 style): response-masked NLL on a HF dataset of (problem, solution) pairs. Entrypoint: `format_train.py`, sbatch wrapper: `slurm/sbatch_format.sh`.
- **Warm-start RL** from another LoRA adapter via `init_from_adapter=…` / `INIT_FROM_ADAPTER=…`.
- **Full resume** from `checkpoints.jsonl` (mid-flight restart) and **legacy load** of full-state checkpoints via `load_checkpoint_path=…`.
- **Small-model (Llama-3B/Qwen) co-resident** training: vLLM and trainer share one GPU.
- **Big-model split-GPU** training: vLLM on GPUs 0..VLLM_TP_SIZE-1, trainer on the rest. Verified end-to-end on gpt-oss-20b on H100 preempt.
- **Curiosity rewards (RND)**, LLM-as-judge (Gemini) reasoning rewards, dataset scheduling (e/h/m warmup vs. main).
- **Evaluation**: `math_evaluation.py` against AIME / math benchmarks via vLLM.

### Not ready / deferred

- **Time-multiplex big-model mode** (`co_resident=False` with vLLM `/sleep`+`/wake_up`): not implemented. Big-model runs currently require enough GPUs to split vLLM and trainer.
- **Async off-policy training** (`max_steps_off_policy != None`): code path exists but is not exercised; defaulted to off.
- **Adapter janitor**: no automatic cleanup of intermediate adapter dirs. Manual `find … | head -n -10 | xargs rm -rf` works (see `slurm/README.md`).
- **gpt-oss harmony format**: SFT/RL on gpt-oss assumes inputs are already harmony-formatted. There is no tooling here to convert non-harmony data.

---

## One-time setup

Done once per machine, not per job.

```bash
cd /home/ksnair/worktrees/slurm

# 1. Initialize the tinker-cookbook submodule (math envs, renderers, supervised helpers).
git submodule update --init --recursive

# 2. Activate the env (clm is the one we've been using; has vLLM 0.20.2 + torch 2.11 + cu129).
export MAMBA_EXE='/home/ksnair/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/ksnair/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate clm
```

For **gpt-oss-20b** specifically, two manual env patches are required (DeepGEMM build + a triton-kernel `.contiguous()` fix). Both are documented at the top of `slurm/README.md` under "Manual env tweaks (gpt-oss / MoE LoRA)". They are env-local; re-apply if the env is recreated.

### Cache layout

`sbatch_format.sh` forces writable per-user caches because `/data/hf_cache/{hub,datasets}` is the community-shared cache and the filelock that both `huggingface_hub` and `datasets` acquire on read needs **write** access — which our user has on the login node but not from compute:

```bash
export HF_HUB_CACHE=/data/hf_cache/ksnair/.hf_hub_cache
export HF_DATASETS_CACHE=/data/hf_cache/ksnair/.hf_datasets_cache
```

This means models and datasets used by the format SFT stage need to be **pre-staged** on the login node into those per-user paths (where you have internet), e.g.:

```bash
export HF_HUB_CACHE=/data/hf_cache/ksnair/.hf_hub_cache
export HF_DATASETS_CACHE=/data/hf_cache/ksnair/.hf_datasets_cache
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('<model>'); AutoTokenizer.from_pretrained('<model>')"
python -c "from datasets import load_dataset; load_dataset('<dataset>', '<config>', split='train')"
```

`sbatch_train.sh` and `sbatch_train_bigmodel.sh` still default to `HF_DATASETS_CACHE=/data/hf_cache/datasets` and `HF_HUB_CACHE=/data/hf_cache/hub`. Those have been working for the RL flow because the math envs route around the locking path that bites format SFT; if you hit the same `PermissionError: ... .lock` from those scripts, apply the same override there.

`LOG_DIR` defaults to `/data/hf_cache/ksnair/CuriousLLMs_logs/<stamp>-<jobid>-<model>` (configurable; see per-stage docs below).

---

## Stage 1 — Format SFT (cold-start)

Teach the LoRA the target output format (chat-template wrapping + `\boxed{answer}` for math) **before** RL kicks in. Stops RL from having to learn formatting from random init. Output is a LoRA adapter directory you then warm-start RL from.

### Submit

```bash
sbatch --export=ALL,\
MODEL_NAME=meta-llama/Llama-3.2-3B,\
DATASET_NAME=AI-MO/NuminaMath-CoT,\
NUM_STEPS=500 \
       slurm/sbatch_format.sh
```

`sbatch_format.sh` defaults to L40S:1, 8h walltime, general partition.

### Env vars

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B` | HF base model id |
| `LORA_RANK` | 32 | Must match stage 2 |
| `DATASET_NAME` | `AI-MO/NuminaMath-CoT` | HF dataset id |
| `DATASET_SPLIT` | `train` | |
| `DATASET_CONFIG` | unset | e.g. `main` for `openai/gsm8k` |
| `PROBLEM_COLUMN` | `problem` | Column with the problem text |
| `SOLUTION_COLUMN` | `solution` | Column with the reference solution |
| `MAX_ROWS` | unset = all | Cap dataset for quick smokes |
| `MAX_SEQ_LEN` | 4096 | Drop rows exceeding this many tokens |
| `BATCH_SIZE` | 8 | Datums per micro-batch |
| `NUM_STEPS` | 500 | Optimizer steps |
| `LEARNING_RATE` | 1e-4 | |
| `SAVE_EVERY` | 100 | Step-interval for intermediate checkpoints |
| `LOG_DIR` | auto | Override to a fixed path if you want |
| `EXTRA_ARGS` | empty | Passed verbatim to `format_train.py` |

### Output

```
<LOG_DIR>/
  format_metrics.jsonl          # per-step loss + response_tokens
  vllm.log                      # vLLM stays idle but logs here
  states/step_NNNNNN/           # intermediate snapshots (every SAVE_EVERY)
  states/format_final/          # ← THE final adapter — point INIT_FROM_ADAPTER here
    adapter_model.safetensors
    adapter_config.json
    optimizer.pt
    meta.json
    rng.pt
```

The slurm output stream will end with a line `FORMAT_FINAL_PATH=<absolute path>` you can grep for.

### Tweaks

- **Faster smoke**: `BATCH_SIZE=2 NUM_STEPS=20 MAX_ROWS=200`.
- **Tighter memory** (e.g. 7B on L40S): `MAX_SEQ_LEN=2048 BATCH_SIZE=4`.
- **Different dataset shape**: `EXTRA_ARGS="problem_column=question solution_column=answer"` for GSM8K, plus `DATASET_CONFIG=main`.

### What the loss looks like

`local_backend/losses.py` `nll_loss` is weighted-mean NLL with `weights` = 0 on prompt tokens, 1 on response tokens (the response mask is built in `format_dataset.py`). Loss should drop sharply in the first ~30 steps then plateau.

If the loss is flat at zero: usually means `weights.sum() == 0` for every row, i.e. tokenization misalignment skipped every row. Check the `skipped: ... tokenizer-misaligned` line in the trainer log — large counts there indicate the chat template / solution concatenation produced different token boundaries than encoding the prompt alone. Try `EXTRA_ARGS="format_template=…"` to override the wrapping or a different `MODEL_NAME`.

---

## Stage 2 — RL (curiosity + PPO)

Group-relative PPO with RND curiosity rewards and optional Gemini reasoning-judge rewards.

### Submit (small model, co-resident)

```bash
sbatch --export=ALL,\
MODEL_NAME=meta-llama/Llama-3.2-3B,\
LORA_RANK=32,\
INIT_FROM_ADAPTER=/data/hf_cache/ksnair/CuriousLLMs_logs/<stage1-run>/states/format_final \
       slurm/sbatch_train.sh
```

### Submit (big model, GPU split)

```bash
sbatch --export=ALL,\
MODEL_NAME=openai/gpt-oss-20b,\
LORA_RANK=16,\
VLLM_TP_SIZE=2,\
TRAINER_NUM_GPUS=1,\
INIT_FROM_ADAPTER=/data/.../states/format_final \
       slurm/sbatch_train_bigmodel.sh
```

`sbatch_train_bigmodel.sh` requests `gpu:A100_80GB:4` by default (4 GPUs: 2 for vLLM TP, 1 for trainer, 1 spare). For preempt H100, override the `--gres` line or copy the script.

### Env vars (both small and big sbatch)

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B` (small) / `openai/gpt-oss-20b` (big) | |
| `LORA_RANK` | 32 (small) / 16 (big) | Trainer rank + `vllm --max-lora-rank` |
| `GROUP_SIZE` | 16 (small) / 8 (big) | Rollouts per problem |
| `GROUPS_PER_BATCH` | 128 (small) / 64 (big) | Problems per batch |
| `LEARNING_RATE` | 7e-5 (small) / 5e-6 (big) | |
| `ENV` | `mixed` | `arithmetic` / `math` / `mixed` / `polaris` / `deepmath` / `gsm8k` |
| `DATASET_SCHEDULE` | `m-m` | warmup-main: `e`=easy/math, `h`=hard/deepmath, `m`=mixed |
| `LOSS_FN` | `ppo` | `ppo` or `importance_sampling` |
| `VLLM_GPU_MEM_FRAC` | 0.45 (small) / 0.85 (big) | vLLM's share |
| `VLLM_TP_SIZE` | 1 (small) / 2 (big) | Tensor parallelism degree |
| `TRAINER_NUM_GPUS` | (big only) 1 | Use `>1` to launch via `accelerate launch` |
| `INIT_FROM_ADAPTER` | unset | Path to a LoRA adapter dir (warm-start from format SFT). Fresh optimizer, batch 0. |
| `LOG_DIR` | auto | |
| `EXTRA_ARGS` | empty | Anything else, passed to `math_train.py` |

### Adapter init priority

When the trainer starts, it picks one of these (in order):

1. **Resume.** If `<LOG_DIR>/checkpoints.jsonl` exists, full state restore from the latest entry. `start_batch` from the manifest. (`behavior_if_log_dir_exists=resume` is already set by the sbatch script.)
2. **`load_checkpoint_path`.** Full state restore (adapter + optimizer + RNG + step counter). `start_batch` stays 0.
3. **`init_from_adapter`.** Adapter weights only. Optimizer freshly initialized, `step_counter=0`. **Use this for chaining stages.**

Specifying both `load_checkpoint_path` and `init_from_adapter` is an error; pick one.

### Output

```
<LOG_DIR>/
  checkpoints.jsonl             # append-only manifest
  metrics.jsonl                 # per-iter training metrics
  states/step_NNNNNN/           # full-state snapshots
  adapters/step_NNNNNN/         # sampler adapters (hot-loaded into vLLM)
  vllm.log
  rnd/                          # RND predictor state
```

---

## Stage 3 — Evaluation

```bash
sbatch --export=ALL,\
MODEL_NAME=meta-llama/Llama-3.2-3B,\
MODEL_PATH=/data/.../adapters/step_NNNNNN \
       slurm/sbatch_eval.sh
```

`math_evaluation.py` evaluates AIME 2024 / Math500 / etc. against the adapter loaded into vLLM. Output goes to `<LOG_DIR>/eval/`.

---

## Chained recipe (the common path)

```bash
cd /home/ksnair/worktrees/slurm

# Stage 1: format cold-start (~1-4h)
J1=$(sbatch --parsable --export=ALL,\
MODEL_NAME=meta-llama/Llama-3.2-3B,\
DATASET_NAME=AI-MO/NuminaMath-CoT,\
NUM_STEPS=500 \
       slurm/sbatch_format.sh)
echo "format SFT job: $J1"

# Wait for it (or just check squeue), then grep the format_final path:
FORMAT_ADAPTER=$(grep "FORMAT_FINAL_PATH=" \
    /data/hf_cache/ksnair/CuriousLLMs_logs/slurm-${J1}.out | tail -1 | cut -d= -f2)
echo "Adapter: $FORMAT_ADAPTER"

# Stage 2: RL warm-started from that adapter (~24h)
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,\
MODEL_NAME=meta-llama/Llama-3.2-3B,\
LORA_RANK=32,\
INIT_FROM_ADAPTER=$FORMAT_ADAPTER \
       slurm/sbatch_train.sh)
echo "RL job: $J2 (will start after $J1 succeeds)"

# Stage 3: eval the RL run (optional)
# Identify the latest RL adapter and submit sbatch_eval.sh.
```

`--dependency=afterok:` makes stage 2 wait for stage 1 to succeed.

---

## CLI reference

### `format_train.py`

chz CLI; pass `field=value` (no `--`).

```bash
python format_train.py \
    model_name=meta-llama/Llama-3.2-3B \
    lora_rank=32 \
    dataset_name=AI-MO/NuminaMath-CoT \
    dataset_split=train \
    problem_column=problem \
    solution_column=solution \
    max_seq_len=4096 \
    batch_size=8 \
    num_steps=500 \
    learning_rate=1e-4 \
    save_every=100 \
    log_path=/data/.../my-format-run \
    base_url=$VLLM_URL
```

Optional: `dataset_config`, `max_rows`, `adam_beta1/2`, `adam_eps`, `weight_decay`, `grad_clip_norm`, `seed`, `wandb_project/entity/name`, `behavior_if_log_dir_exists`.

### `math_train.py`

```bash
python math_train.py \
    model_name=meta-llama/Llama-3.2-3B \
    lora_rank=32 \
    group_size=16 \
    groups_per_batch=128 \
    learning_rate=7e-5 \
    env=mixed \
    dataset_schedule=m-m \
    loss_fn=ppo \
    base_url=$VLLM_URL \
    log_path=/data/.../my-rl-run \
    init_from_adapter=/data/.../format_final     # warm-start
```

All RL-specific knobs live in `math_train.py`'s `CLIConfig` (RND coefficients, curiosity warmup, reasoning rewards, dataset sub-sizes). See the file directly — `chz` reflects all fields with defaults.

### `math_evaluation.py`

```bash
python math_evaluation.py \
    model_name=meta-llama/Llama-3.2-3B \
    model_path=/data/.../adapters/step_000050 \
    base_url=$VLLM_URL \
    log_path=/data/.../eval-run
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RuntimeError: LocalServiceClient requires vllm_url` | Running `format_train.py` / `math_train.py` outside sbatch with no vLLM | Launch vLLM first (`source slurm/launch_vllm.sh`) or sbatch the wrapper |
| `LoRA rank mismatch loading <path>: adapter r=16, trainer lora_rank=32` | Stage 1 trained at one rank, stage 2 set a different rank | Make `LORA_RANK` match across stages |
| `ValueError: Both load_checkpoint_path and init_from_adapter are set` | Both flags passed | Pick one (see "Adapter init priority" above) |
| `RuntimeError: No usable rows in <dataset>` | All rows dropped (empty / too long / tokenizer-misaligned) | Check `problem_column`/`solution_column`, raise `max_seq_len`, try a different `model_name` (chat template matters) |
| `vllm.log` shows `DeepGEMM backend is not available` | gpt-oss path on a fresh env | Apply the DeepGEMM build patch in `slurm/README.md` |
| `AssertionError: inputs.is_contiguous()` in `lora_shrink_op.py` | gpt-oss LoRA on un-patched vLLM | Apply the `.contiguous()` patch in `slurm/README.md` |
| GptOss complains "sdpa not supported" | Default attn is sdpa | sbatch scripts already export `LOCAL_ATTN_IMPL=eager`; set it if running interactively |
| Trainer OOM during forward | vLLM + trainer fighting for memory | Lower `VLLM_GPU_MEM_FRAC` or set `LOCAL_TRAINER_GPU_MEM_FRAC` env var (sets `torch.cuda.set_per_process_memory_fraction`) |
| `metrics.jsonl` shows `loss=0` for most batches | Group-centered advantages: every group's reward is uniform | This is expected when rewards don't differentiate within a group (e.g. all-correct or all-wrong). Surface the unmasked loss via `EXTRA_ARGS="remove_constant_reward_groups=False"` to keep gradient signal from uniform groups |

---

## Tests

```bash
cd /home/ksnair/worktrees/slurm
export PYTHONPATH="./tinker-cookbook:${PYTHONPATH:-}"
micromamba activate clm
pytest tests/ -q
```

- `tests/test_nll_loss.py` — CPU-only, verifies the NLL loss math and dispatch.
- `tests/test_init_from_adapter.py` — needs CUDA, downloads `Qwen/Qwen2.5-0.5B-Instruct`. Verifies warm-start loads weights from disk (vs. random re-init) and leaves a fresh optimizer.
- `tests/test_local_backend_smoke.py` — pre-existing smoke tests.

---

## File map

| Path | Purpose |
| --- | --- |
| `format_train.py` | Stage 1 entrypoint — format SFT cold-start (NLL on HF dataset) |
| `math_train.py` | Stage 2 entrypoint — RL (curiosity + PPO/IS) |
| `math_evaluation.py` | Stage 3 entrypoint — AIME / math eval |
| `curiosity_train.py` | Core RL loop (called from `math_train.py`) |
| `local_backend/training_client.py` | HF + PEFT + AdamW LoRA trainer (PPO / IS / NLL) |
| `local_backend/sampling_client.py` | vLLM HTTP client |
| `local_backend/service_client.py` | Tinker-shaped service shim |
| `local_backend/weight_sync.py` | Pushes adapter dirs to vLLM via `/v1/load_lora_adapter` |
| `local_backend/losses.py` | `ppo_loss`, `importance_sampling_loss`, `nll_loss` |
| `local_backend/format_dataset.py` | HF dataset → response-masked `tinker.Datum` list |
| `local_backend/vllm_proc.py` | `subprocess.Popen` wrapper for vLLM |
| `slurm/sbatch_format.sh` | SLURM wrapper for stage 1 |
| `slurm/sbatch_train.sh` | SLURM wrapper for stage 2 (small models, co-resident) |
| `slurm/sbatch_train_bigmodel.sh` | SLURM wrapper for stage 2 (big models, GPU split) |
| `slurm/sbatch_eval.sh` | SLURM wrapper for stage 3 |
| `slurm/launch_vllm.sh` | Sourced helper that boots vLLM with the right flags |
| `slurm/README.md` | Operator notes + DeepGEMM / lora_shrink_op env patches |
| `tinker-cookbook/` | Submodule: math envs, renderers, RL types, supervised helpers |
