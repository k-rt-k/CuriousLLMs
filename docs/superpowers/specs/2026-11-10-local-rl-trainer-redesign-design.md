# Local RL trainer redesign — design

Date: 2026-11-10. Branch base: `slurm` (HEAD 6b4b8e6).

## Intent

CuriousLLMs runs curiosity-driven RL (RND novelty rewards, optional entropy bonus, KL penalty) on math
datasets. Experiments run on Tinker and on Babel (CMU SLURM cluster). Babel runs must be comparable to
Tinker runs, but **only the hyperparameters and the loss term have to match** — the RL loop on Babel may
be rewritten freely and need not reuse tinker-cookbook's loop or Tinker's client API.

The current Babel path (`local_backend/`, a Tinker-API-compatible shim) works for small models but:
replicates the full base model on every trainer GPU (no sharding), gets KL logprobs from vLLM
`prompt_logprobs` (crashed vLLM with a CUDA device-side assert in GPU smoke job 11108051), runs
micro-batches one datum at a time, and does not match Tinker's LoRA placement (see Constraints).

## Goals

- Train **Qwen3-8B** and **gpt-oss-20b** with LoRA RL on Babel, scaling to "enough GPUs/nodes to run".
- Match Tinker in hyperparameters and loss (exact list under Constraints).
- **Optional FSDP** sharding of the trainer, so gpt-oss-20b can run on the plentiful 48 GB cards
  (L40S / A6000 / 6000 Ada) when 80 GB+ queues are long. Off by default; on via a flag.
- Fewer checkpoints than today.
- Survive preemption (Babel is preempt-only).

## Non-goals

- Bit-identical or step-identical results with Tinker (different kernels, sampling RNG).
- Keeping the Tinker client API on the local side, or keeping `curiosity_train.py` shared between backends.
- Full fine-tuning, models larger than gpt-oss-20b (e.g. gpt-oss-120b), throughput tuning beyond
  "batched micro-batches".
- Async/off-policy training.

## Anti-goals (would be a failure even if it runs)

- A silent hyperparameter or loss mismatch with Tinker. Every known mismatch must be documented and
  logged at startup.
- KL / logprob numbers computed by different numerics than the gradient sees without saying so.
- A run that loses more than `save_every` steps on preemption, or that cannot resume.

## Constraints

**Hyperparameter / loss parity with Tinker (binding):**
- LoRA rank 32 default (CLI-configurable), `lora_alpha = 32` regardless of rank, dropout 0,
  HF-`peft` init (A uniform, B zero). Same LR for A and B.
- LoRA placement = Tinker defaults `train_attn=True, train_mlp=True, train_unembed=True`
  (verified in installed `tinker` 0.19.0 `create_lora_training_client`; the Tinker docs state MLP includes
  MoE layers). So: attention projections, MLP projections, MoE expert weights, and `lm_head`.
  - gpt-oss experts are 3D parameters, so they need PEFT `target_parameters`
    (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`), not `target_modules`.
  - **Known mismatch, documented:** PEFT/vLLM give each expert its own rank-r LoRA. Tinker's
    per-expert rank for production MoE LoRA is undocumented (the LoRA blog's experiment used
    total rank / active experts). Log this at startup for MoE models.
  - **gpt-oss `lm_head` fallback (user decision):** vLLM 0.20.2's gpt-oss model class does not register
    `lm_head` in `embedding_modules`, so vLLM rejects an adapter containing it. Try registering it via a
    vLLM general plugin (as Qwen3 does). If that is fragile or fails verification, gpt-oss runs train
    **without** `lm_head` LoRA; document the mismatch in the docs and log it at startup.
- Optimizer: Adam with β1 0.9, β2 0.95, eps 1e-8, LR from config (what `curiosity_train.optim_step`
  passes to Tinker today). No weight decay (confirm against Tinker's AdamParams default when building).
- Loss: Tinker's `importance_sampling` and `ppo` (clip 0.8/1.2) as **token-sum** over response tokens
  (no per-token or per-sequence mean). Multi-GPU gradients are SUMMED across ranks, never averaged.
- Advantages: group-centered rewards (reward − group mean), no std normalization; KL penalty folded into
  advantages as in `tinker_cookbook.rl.metrics.incorporate_kl_penalty` (coef, discount, `kl_reference`
  base|init); entropy bonus as currently on `slurm` (added to rewards before advantages; RND correctness
  judged on pre-bonus rewards).
- Sampling temperature 1.0, same `max_tokens`, `group_size`, `groups_per_batch`, renderer/chat template
  as the Tinker run config.

**Platform:**
- Babel `preempt` partition only, `preempt_qos` caps 24 GPUs per user; jobs must handle SIGTERM/requeue.
- GPU pool: ~50 L40S nodes (48 GB), A6000 / 6000 Ada (48 GB), RTX PRO 6000 (96 GB, ~20 nodes),
  A100-80GB (6 nodes), H100 (3), H200 (1).
- vLLM 0.20.2 in the `clm` env (with the two documented env patches for gpt-oss). vLLM 0.20.2 code
  supports MXFP4 MoE LoRA on CUDA (Triton `UnfusedOAITritonExperts`, else Marlin) — present in source,
  **not yet verified on GPU**.
- Logs/checkpoints on `/data/hf_cache/ksnair/CuriousLLMs_logs/`.

## Decided parts of the how

- **Rewrite** the local RL loop as a lean, purpose-built loop (not TRL / verl / OpenRLHF / Unsloth).
  Reasons: TRL syncs LoRA by merging into full weights (awkward with MXFP4 experts) and neither TRL nor
  verl offers Tinker's token-sum loss without subclassing; vLLM already serves the needed LoRA.
  *(Claude's call; user agreed rewriting is fine.)*
- **Sampling:** vLLM OpenAI server(s), LoRA hot-swapped from a saved adapter dir after each optimizer
  step (keep the multi-server fan-out/load balancing from `slurm`).
- **Trainer:** HF transformers + PEFT LoRA, bf16, batched/padded (or packed) micro-batches.
  - Default: data parallel with full model replicas, SUM all-reduce of LoRA grads (current approach).
  - Optional `fsdp=True`: FSDP2 `fully_shard` over the trainer ranks (works across nodes), frozen base
    weights sharded, LoRA params trained; gradient reduction must still be a SUM.
- **KL / reference logprobs computed in the trainer** (adapter disabled for `kl_reference=base`, a
  stored copy of the initial adapter for `init`), not via vLLM `prompt_logprobs`. Same for the
  post-update KL metric.
- **Checkpoints:** save adapter + optimizer + RNG + step every `save_every` steps (default larger than
  today's per-step save), keep only the last K (default 2), plus an emergency save on SIGTERM before
  exit. Resume from the latest.
- **Placement:** vLLM on its own GPU(s); trainer on the remaining GPUs/nodes. Qwen3-8B fits replicated
  on 48 GB cards; gpt-oss-20b fits replicated on 80 GB+ cards and needs `fsdp=True` on 48 GB cards.
- **Tinker path stays as-is** (`curiosity_train.py` + Tinker); only the shared pieces it needs for parity
  (reward/entropy/RND code, configs) are reused or kept in sync.

## First step before building on it: verification spike (throwaway)

On one GPU node, with gpt-oss-20b and Qwen3-8B: save a PEFT adapter covering attention + MLP/experts
(+ `lm_head`), load it into vLLM 0.20.2, and check vLLM logprobs ≈ HF trainer logprobs for the adapted
model. Outcome decides: expert LoRA usable as designed; whether the gpt-oss `lm_head` plugin works or
the fallback applies.

## Resolved after builder check (Claude's calls unless marked)

- **If vLLM cannot serve gpt-oss expert LoRA (user decision):** spike fails or logprobs disagree →
  ship Qwen3-8B first, then support gpt-oss by **merging LoRA into the base weights and reloading
  the merged model in vLLM** each step (slower; keeps full Tinker LoRA placement). No attention-only
  fallback for experts.
- **Feature scope of the new loop (user decision):** RND curiosity (incl. `curiosity_warmup_batches`),
  Gemini LLM-as-judge reasoning rewards, PPO and importance-sampling losses, in-loop evaluation
  (`eval_every`), KL penalty (`kl_reference` base|init), entropy bonus, `init_from_adapter`.
  Out of scope until a run needs them: `dataset_schedule`, `remove_constant_reward_groups`,
  `num_substeps` > 1. Reference config: `logs/rnd_training/Llama-3B/config.json` (PPO, LR 7e-5,
  rank 32, RND + judge on, warmup 20, eval every 20). SemanticRND's encoder runs on rank 0's GPU.

- **Which logprobs feed the loss and bonus:** the importance ratio's denominator and the entropy bonus
  use **sampler (vLLM) logprobs**, exactly as Tinker does. Only KL/reference logprobs move to the
  trainer. The sampler-vs-trainer numerics gap is inherent to Tinker too; log mean |Δ| between sampler
  and trainer logprobs on the first batch of each step as a metric (no hard threshold).
- **gpt-oss trainer weights:** the trainer dequantizes MXFP4 experts to bf16 (frozen); vLLM serves
  MXFP4 + LoRA. Spike pass criterion: per-token mean |logprob_vLLM − logprob_HF| ≤ 0.05 and LoRA
  effect visible (adapted-vs-base mean |Δ| clearly above that), for both models.
- **Checkpoint contents:** one atomic directory per save: adapter, optimizer, RNG, step, RND predictor +
  buffer, dataloader position, wandb run id. No need to load old `local_backend` checkpoints.
  `init_from_adapter` / `kl_reference=init` accept PEFT-format adapter dirs (including ones exported
  from Tinker).
- **FSDP scope:** FSDP is built multi-node-capable (torchrun across nodes), but the first deliverable is
  verified on a single 8-GPU node (gpt-oss-20b on 48 GB cards); multi-node verification follows.
- **`lm_head` / `train_unembed`:** existing Tinker runs never pass `train_*` arguments, so they used
  Tinker's defaults (all True). Match those.

## Left to the builder

- Module/file layout of the new loop and how much of `local_backend/` is reused.
- Padding vs packing, micro-batch sizing, activation checkpointing.
- FSDP wrapping policy and mixed-precision settings (keep grads/optimizer state fp32 for LoRA params).
- Process launch (torchrun / srun) and how rank 0 coordinates sampling vs training.
- Exact default `save_every` and checkpoint retention K.
