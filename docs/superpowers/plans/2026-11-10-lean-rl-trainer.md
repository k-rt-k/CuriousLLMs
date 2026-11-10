# Lean Local RL Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A purpose-built local RL loop (`lean_rl/`) that trains Qwen3-8B and gpt-oss-20b with LoRA on Babel, matching Tinker's hyperparameters and loss, with optional FSDP sharding and lean checkpoints.

**Architecture:** SPMD over torchrun ranks. Rank 0 does rollouts (tinker-cookbook envs + `TinkerTokenCompleter` over the existing `local_backend.LocalSamplingClient` → vLLM), reward shaping (RND, judge, entropy bonus — extracted from `curiosity_train.py` into a shared module), advantages, and KL (cookbook `incorporate_kl_penalty` fed by trainer-computed reference logprobs); it broadcasts the batch; every rank runs a batched token-sum PPO/IS step on its shard (replicated + SUM all-reduce, or FSDP2 with divide factor 1). Rank 0 saves the adapter and hot-swaps it into vLLM.

**Tech Stack:** Python 3.11, torch 2.11 (FSDP2 `fully_shard`), transformers 4.57.6, peft 0.18.1 (`target_parameters` for 3D MoE experts), vLLM 0.20.2, tinker 0.19.0 (data types only), tinker-cookbook (envs, rollouts, advantages, KL, ml_log), chz, SLURM/torchrun.

**Spec:** `docs/superpowers/specs/2026-11-10-local-rl-trainer-redesign-design.md`

## Global Constraints

- LoRA: rank from config (default 32), `lora_alpha=32` regardless of rank, `lora_dropout=0.0`, PEFT default init; placement = attention + MLP + MoE experts + `lm_head` (Tinker defaults `train_attn/train_mlp/train_unembed=True`). gpt-oss experts via PEFT `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`.
- gpt-oss `lm_head`: included only if the vLLM plugin (Task 2) passes the spike; otherwise excluded, and the mismatch is documented and logged at startup.
- Every startup logs known Tinker mismatches (MoE per-expert rank; gpt-oss `lm_head` if excluded) as `WARNING [tinker-parity] ...`.
- Optimizer: `torch.optim.AdamW` with lr/beta1/beta2/eps/weight_decay taken field-by-field from the `tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)` object (its other fields at their tinker defaults, read at runtime — never hard-coded). tinker 0.19 `AdamParams` fields: learning_rate, beta1, beta2, eps, weight_decay (default 0.0), grad_clip_norm (default 0.0 = off); when `grad_clip_norm > 0`, clip the global LoRA grad norm (after the SUM reduction; FSDP-aware) before the step.
- Loss: `importance_sampling` = −Σ_resp ratio·A; `ppo` = −Σ_resp min(ratio·A, clip(ratio, 0.8, 1.2)·A); ratio = exp(logp_new − logp_sampler). **Token SUM** over response tokens across the whole batch; no mean. Multi-rank gradients SUMMED, never averaged (FSDP2: `set_gradient_divide_factor(1.0)`).
- Importance-ratio denominator and entropy bonus use **sampler (vLLM) logprobs**. KL/reference logprobs come from the trainer.
- Advantages: cookbook `compute_advantages` (group-centred, no std norm); KL via cookbook `incorporate_kl_penalty` unchanged.
- Sampling temperature 1.0; same `max_tokens`, `group_size`, `groups_per_batch`, renderer as the Tinker config.
- Feature scope: RND (+ `curiosity_warmup_batches`), Gemini judge, PPO & IS, `eval_every`, KL (`kl_reference` base|init), entropy bonus, `init_from_adapter`. Not in scope: `dataset_schedule`, `remove_constant_reward_groups`, `num_substeps>1`, async/off-policy.
- Only rank 0 logs, talks to vLLM, runs RND/judge, writes checkpoints.
- The Tinker path (`curiosity_train.py` + Tinker) keeps its behaviour; Task 3 only extracts code from it.
- SLURM: `--partition=preempt --qos=preempt_qos --requeue --open-mode=append`, ≤24 GPUs/user, logs under `/data/hf_cache/ksnair/CuriousLLMs_logs/`. All `slurm/*.sh` are zsh; Ruling-D `REPO_DIR` pattern; `TRITON_CACHE_DIR` exported in the parent shell as in existing scripts.
- Tests run on Babel CPU via `.superpowers/sdd/<plan-workspace>/babel-test.sh` (rsync + pytest in `clm`); GPU checks via `modal_tests/gpu_tests.py` or a Babel GPU job, run by the controller.

## Review Focus

1. A batch where every datum lands on one rank (fewer datums than ranks) — the empty rank must still join every collective (FSDP forward/backward and all-reduce) and the step must equal the 1-rank step. → Task 5 `test_fewer_datums_than_ranks`.
2. Preemption mid-step (SIGTERM between rollout and optimizer step) — must save a consistent checkpoint (no half-applied step) and resume to the same step count. → Task 6 `test_sigterm_mid_step_resumes_consistently`.
3. Padding: right-padded batches with very different lengths must give the same loss/grads as unpadded per-datum computation. → Task 4 `test_padded_batch_matches_per_datum`.
4. `kl_reference=init` with `init_from_adapter` — the reference must be the initial adapter, not the live one, and must not change after optimizer steps. → Task 4 `test_init_reference_frozen`.
5. Retention: with keep=2, older checkpoints are deleted only after the new one is fully written (crash during save never leaves zero valid checkpoints). → Task 6 `test_crash_during_save_keeps_previous`.

---

### Task 1: LoRA placement matching Tinker

**Files:**
- Create: `lean_rl/__init__.py` (empty), `lean_rl/lora.py`
- Test: `tests/lean_rl/test_lora.py` (+ `tests/lean_rl/__init__.py` empty if needed by pytest config; reuse `tests/conftest.py`)

**Interfaces:**
- Produces:
  - `@dataclass LoraPlacement: target_modules: list[str]; target_parameters: list[str]; includes_lm_head: bool; mismatches: list[str]`
  - `lora_placement(model: PreTrainedModel, *, train_unembed: bool = True, gpt_oss_lm_head: bool = False) -> LoraPlacement`
  - `make_lora_config(placement: LoraPlacement, rank: int) -> peft.LoraConfig` (alpha 32, dropout 0.0)
  - `log_parity_warnings(placement: LoraPlacement, model_type: str) -> None` (one `WARNING [tinker-parity] ...` per mismatch)

- [ ] **Step 1: Write failing tests** with tiny random models built in-test (no network):
  - `test_dense_targets_all_linear_plus_lm_head` (tiny `LlamaConfig` and tiny `Qwen3Config`): `target_modules` ⊇ {q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head}; `target_parameters == []`; `includes_lm_head`; `mismatches == []`.
  - `test_gpt_oss_experts_via_target_parameters` (tiny `GptOssConfig`, e.g. 2 layers, 4 experts, hidden 64): `target_parameters == ["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`; attention projections in `target_modules`; router NOT targeted; with default `gpt_oss_lm_head=False`, `includes_lm_head is False` and `mismatches` contains a string mentioning `lm_head`; every MoE model's `mismatches` contains a string mentioning `per-expert rank`.
  - `test_peft_model_trains_expected_params`: `get_peft_model(model, make_lora_config(p, 8))` → every trainable param name contains `lora_`; for gpt-oss some trainable names contain `experts`; for dense one contains `lm_head`; `lora_alpha == 32`, `lora_dropout == 0.0`.
  - `test_train_unembed_false_excludes_lm_head`.
- [ ] **Step 2:** Run `babel-test.sh tests/lean_rl/test_lora.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `lean_rl/lora.py`. Detect linear layers by walking `named_modules()` for `nn.Linear` leaf names excluding the router (`mlp.router`) and `lm_head` unless `train_unembed`; detect 3D experts by `model.config.model_type == "gpt_oss"`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `lean_rl/__init__.py lean_rl/lora.py tests/lean_rl/`.

### Task 2: Expert-LoRA serving spike tooling + gpt-oss lm_head vLLM plugin

**Files:**
- Create: `lean_rl/vllm_plugin/pyproject.toml`, `lean_rl/vllm_plugin/lean_rl_vllm_plugin/__init__.py`
- Create: `lean_rl/tools/spike_lora_serving.py`
- Modify: `slurm/README.md` ("Manual env tweaks": install the plugin with `pip install -e lean_rl/vllm_plugin` and enable via `VLLM_PLUGINS=lean_rl_gpt_oss_lm_head`)

**Interfaces:**
- Consumes: Task 1 `lora_placement`, `make_lora_config`.
- Produces: vLLM general plugin entry point `vllm.general_plugins: lean_rl_gpt_oss_lm_head = lean_rl_vllm_plugin:register`; `register()` sets `GptOssForCausalLM.embedding_modules = {"lm_head": "output_embeddings"}` (idempotent).
- Produces: CLI `python -m lean_rl.tools.spike_lora_serving --model M --vllm-url U --out DIR [--with-lm-head]` that exits 0 iff PASS and prints a one-line JSON verdict `{"model","with_lm_head","mean_abs_vs_hf","adapter_effect","pass"}`.

- [ ] **Step 1:** Write `tests/lean_rl/test_vllm_plugin.py::test_register_sets_embedding_modules` (skip if vLLM import fails): after `register()`, `GptOssForCausalLM.embedding_modules["lm_head"] == "output_embeddings"`; calling twice is harmless.
- [ ] **Step 2:** Run → FAIL. Implement plugin package. Run → PASS.
- [ ] **Step 3:** Implement the spike CLI: build PEFT model per Task 1 on cuda (bf16), set every `lora_B` to `randn*0.02` (seed 0), save adapter, `local_backend.weight_sync.load_lora([url], "spike", dir)`, then for 4 fixed prompts compare `LocalSamplingClient(..., adapter_name="spike").compute_logprobs` against HF teacher-forced logprobs. PASS iff mean |Δ| ≤ 0.05 AND adapted-vs-base mean |Δ| (HF) ≥ 0.10. `bash -n`/py_compile only on CPU.
- [ ] **Step 4:** Commit. **Controller gate (GPU, Ruling C style):** run the spike for Qwen/Qwen3-8B (with lm_head) and openai/gpt-oss-20b (with and without `--with-lm-head`, plugin installed). Record verdicts in the ledger. Decision rule (spec): gpt-oss experts FAIL → Tasks 3–8 proceed for Qwen3 only, then Task 9. gpt-oss `lm_head` FAIL → `gpt_oss_lm_head=False` stays default (already the default).

### Task 3: Extract reward shaping into a shared module

**Files:**
- Create: `lean_rl/shaping.py`
- Modify: `curiosity_train.py` (`prepare_minibatch` body from "Compute trajectory metrics" through the end of the SEMANTIC_RND block → one call)
- Test: `tests/lean_rl/test_shaping.py`

**Interfaces:**
- Produces: `async def shape_rewards(env_group_builders_P, trajectory_groups_P, tokenizer, *, i_batch: int, gemini_judge, reasoning_reward_coef: float, novelty_model, rnd_buffer, intrinsic_reward_coef: float, rnd_update_steps: int, rnd_minibatch_size: int, penalize_incorrect_novelty: bool, correctness_threshold: float, curiosity_warmup_batches: int, entropy_bonus_coef: float) -> dict[str, float]` — mutates `final_rewards_G` in place exactly as `prepare_minibatch` does today and returns the same metrics keys. `compute_entropy_bonus`/`apply_entropy_bonus` move here; `curiosity_train` re-imports them (names stay importable from `curiosity_train`).

- [ ] **Step 1:** Write `test_shape_rewards_matches_prepare_minibatch_pre_refactor`: build 2 groups × 4 trajectories with known rewards and `maybe_logprobs`; run the **pre-refactor** `curiosity_train.prepare_minibatch` path with judge/RND disabled and `entropy_bonus_coef=0.1`, `kl_penalty_coef=0` (capture `final_rewards_G` and metrics), then `shape_rewards` on a deep copy; assert equal rewards (exact) and equal metric keys/values. Also `test_rnd_path_called_with_pre_bonus_rewards` using a stub novelty model that records `extrinsic_rewards` (asserts pre-bonus values) — move the existing equivalent test from `tests/test_entropy_bonus.py` if present.
- [ ] **Step 2:** Run → FAIL. Implement extraction (cut/paste, no behaviour change). Run → PASS; run the whole `tests/` → still green.
- [ ] **Step 3:** Commit.

### Task 4: Single-process batched trainer

**Files:**
- Create: `lean_rl/trainer.py`, `lean_rl/losses.py`
- Test: `tests/lean_rl/test_trainer.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `lean_rl/losses.py`: `token_sum_loss(loss_fn: Literal["ppo","importance_sampling"], new_logp: Tensor[B,T], sampler_logp: Tensor[B,T], adv: Tensor[B,T], mask: Tensor[B,T]) -> Tensor` (scalar, SUM).
  - `class Trainer` in `lean_rl/trainer.py`:
    - `__init__(self, base_model: str, *, rank: int, device: str, dtype: torch.dtype = torch.bfloat16, train_unembed: bool = True, gpt_oss_lm_head: bool = False, dist: "DistInfo | None" = None, fsdp: bool = False, micro_batch_tokens: int = 16384)`
    - `logprobs(self, data: list[tinker.Datum], which: Literal["policy","base","init"]) -> list[Tensor]` (per datum, aligned to `target_tokens`, fp32 CPU)
    - `train_step(self, data: list[tinker.Datum], loss_fn: str, adam: tinker.AdamParams) -> dict[str, float]` (metrics: `loss`, `response_tokens`, `grad_norm`, `logprob_gap/sampler_vs_trainer_mean_abs` = mean over response tokens of |trainer policy logp − sampler logp| from the step's own forward pass, before the update)
    - `save_adapter(self, path: str) -> None` (rank-0 PEFT dir; gathers under FSDP)
    - `snapshot_init_adapter(self) -> None` (copy of current LoRA weights used for `which="init"`)
    - `load_adapter(self, path: str) -> None`, `state_dict_for_ckpt(self) -> dict`, `load_ckpt_state(self, d: dict) -> None`
    - `as_ref_client(self, which) -> RefLogprobClient` with `async compute_logprobs_async(model_input) -> list[float|None]` (len = tokens, element 0 None) — duck-types the sampling client for cookbook `incorporate_kl_penalty` / `compute_post_kl`.
- Datum convention: `loss_fn_inputs` keys `target_tokens`, `logprobs` (sampler), `advantages`, `mask` exactly as cookbook `assemble_training_data` produces.

- [ ] **Step 1: Failing tests** (tiny Llama from `tests/conftest.py::tiny_model_dir`, fp32, CPU):
  - `test_token_sum_matches_local_backend`: same datums/seed → `Trainer.train_step` LoRA grads equal `local_backend.LocalTrainingClient` grads (atol 1e-6) for both loss fns — the existing backend already implements Tinker's token-sum semantics.
  - `test_padded_batch_matches_per_datum` (Review Focus 3): lengths [3, 40, 7]; batched with `micro_batch_tokens=10_000` vs `micro_batch_tokens=1` (forces one datum per micro-batch) → equal loss and grads (atol 1e-6).
  - `test_logprobs_base_vs_policy`: before any step, `policy == base` (lora_B zero); after one step with nonzero advantages, they differ; `base` unchanged across steps.
  - `test_init_reference_frozen` (Review Focus 4): `load_adapter(p)`; `snapshot_init_adapter()`; two steps; `logprobs(which="init")` equals the logprobs of a fresh trainer loaded from `p`.
  - `test_ref_client_contract`: `as_ref_client("base").compute_logprobs_async` returns len == tokens, `[0] is None`; feeding it to cookbook `incorporate_kl_penalty` with `kl_penalty_coef=0.1` at step 0 gives `kl_policy_base ≈ (sampler − base)` mean.
  - `test_adam_hparams_from_tinker`: optimizer param_group lr/betas/eps/weight_decay equal the `tinker.AdamParams` fields; with `grad_clip_norm=1e-3` the applied update matches a manual clip-then-AdamW step.
  - `test_logprob_gap_metric`: when datums' sampler logprobs equal the trainer's policy logprobs, the gap metric is < 1e-6; shifting sampler logprobs by +0.5 gives 0.5.
- [ ] **Step 2:** Run → FAIL. Implement: right-pad micro-batches up to `micro_batch_tokens`; loss = Σ over micro-batches of `token_sum_loss` (no normalisation); `which="base"` uses `with model.disable_adapter()`; `which="init"` swaps in the snapshot LoRA tensors under `torch.no_grad()` and restores. Run → PASS.
- [ ] **Step 3:** Commit.

### Task 5: Multi-rank: replicated DP and optional FSDP2

**Files:**
- Create: `lean_rl/dist.py`
- Modify: `lean_rl/trainer.py`
- Test: `tests/lean_rl/test_trainer_dist.py` (mp.spawn harness in the style of `tests/test_distributed_trainer.py`, gloo CPU; `DIST_TEST_DEVICE=cuda` runs on GPUs)

**Interfaces:**
- Produces in `lean_rl/dist.py`: `@dataclass DistInfo: rank: int; world_size: int; local_rank: int; is_main: bool`; `init_dist() -> DistInfo | None` (torchrun env; NCCL on CUDA else gloo; timeout `$LEAN_DIST_TIMEOUT_S` default 14400); `broadcast_object(obj, src=0) -> obj`; `shard(data: list, lengths: list[int], dist: DistInfo | None) -> list` (reuse `local_backend.distributed.shard_by_tokens`).
- `Trainer.train_step` on rank r trains on `shard(data)`; replicated mode SUM-all-reduces LoRA grads; `fsdp=True` applies `fully_shard` per decoder layer + root with `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` (fp32 in tests), LoRA params fp32, and `set_gradient_divide_factor(1.0)`.
- `Trainer.logprobs` under multi-rank: every rank computes its shard, results gathered to rank 0 in input order (all ranks must call it).

- [ ] **Step 1: Failing tests** (each for `fsdp in [False, True]`):
  - `test_two_ranks_match_one_rank`: 2-rank step == 1-rank step (state & loss, atol 1e-5).
  - `test_grad_is_sum_not_mean`: 2-rank grads == 1-rank grads (not half).
  - `test_fewer_datums_than_ranks` (Review Focus 1): 1 datum, 2 ranks → equals 1-rank.
  - `test_logprobs_gathered_in_order`: 5 datums, 2 ranks → rank-0 result equals 1-rank result, same order.
  - `test_save_adapter_under_fsdp_loads_in_single_rank`: adapter saved by 2-rank FSDP loads into a 1-rank trainer with identical logprobs.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS (CPU). GPU/NCCL run deferred to controller.
- [ ] **Step 3:** Commit.

### Task 6: Lean checkpoints with retention and SIGTERM save

**Files:**
- Create: `lean_rl/checkpoint.py`
- Test: `tests/lean_rl/test_checkpoint.py`

**Interfaces:**
- Produces:
  - `@dataclass LoopState: step: int; data_position: int; wandb_run_id: str | None`
  - `save_checkpoint(log_path: str, step: int, trainer: Trainer, loop_state: LoopState, rnd_state: dict | None, keep: int = 2) -> str` — writes `ckpt/step_{step:06d}.tmp/` then `os.replace` to `ckpt/step_{step:06d}/`; contains `adapter/`, `optimizer.pt`, `rng.pt` (torch, cuda, numpy, python), `loop_state.json`, `rnd.pt`; deletes older than the newest `keep` only after the rename.
  - `load_latest(log_path: str) -> tuple[str, LoopState] | None` (ignores `.tmp` dirs).
  - `class PreemptionFlag` with `install()` (SIGTERM/SIGUSR1 handler sets `.requested`), used by the loop to save at the next step boundary and exit 0.

- [ ] **Step 1: Failing tests:** `test_roundtrip_restores_identical_training` (save at step 2, load into fresh trainer, one more step equals uninterrupted 3 steps, atol 1e-6); `test_retention_keeps_last_k` (save steps 1..4, keep=2 → only 3,4); `test_crash_during_save_keeps_previous` (Review Focus 5: monkeypatch `os.replace` to raise → step_N-1 still loadable, `.tmp` ignored); `test_sigterm_mid_step_resumes_consistently` (Review Focus 2: subprocess runs a 5-step toy loop using `PreemptionFlag`, SIGTERM after step-2 rollout; resume finishes; final step count 5 and state equals uninterrupted run).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit.

### Task 7: The RL loop, config, and entrypoint

**Files:**
- Create: `lean_rl/config.py`, `lean_rl/loop.py`, `lean_rl/train.py` (`python -m lean_rl.train key=value ...`, chz)
- Test: `tests/lean_rl/test_loop_e2e.py`

**Interfaces:**
- Consumes: Tasks 1, 3–6; `math_train.get_dataset_builder`; cookbook `do_group_rollout`, `TinkerTokenCompleter`, `compute_advantages`, `assemble_training_data`, `incorporate_kl_penalty`, `compute_post_kl`, `ml_log`; `local_backend.LocalSamplingClient`, `local_backend.weight_sync.sync_lora_to_vllm`; `llm_as_a_judge.GeminiJudge`; RND setup as in `curiosity_train.main` (SemanticRND encoder on rank 0's GPU).
- Produces: `@chz.chz class LeanRLConfig` with fields: `model_name, env, group_size, groups_per_batch, learning_rate, max_tokens, lora_rank=32, loss_fn="ppo", renderer_name=None, seed=0, kl_penalty_coef=0.0, kl_discount_factor=0.0, kl_reference="base", compute_post_kl=False, entropy_bonus_coef=0.0, use_rnd_curiosity, use_reasoning_rewards, reasoning_reward_coef, curiosity_warmup_batches, (RND fields mirrored from math_train.CLIConfig with the same names and defaults), eval_every=20, save_every=10, keep_checkpoints=2, init_from_adapter=None, fsdp=False, micro_batch_tokens=16384, gpt_oss_lm_head=False, vllm_urls=None, log_path, wandb_project=None, num_batches=None`; `async def run(cfg: LeanRLConfig) -> None`.
- Per step (rank 0 unless noted): rollouts → `shape_rewards` → advantages → `assemble_training_data` → KL via `incorporate_kl_penalty(data, trainer.as_ref_client(cfg.kl_reference), ...)` (**all ranks** call `trainer.logprobs`; rank 0 broadcasts data first) → broadcast data → **all ranks** `train_step` → `save_adapter` → `sync_lora_to_vllm` → optional `compute_post_kl` (trainer policy client) → eval if `step % eval_every == 0` (cookbook evaluators with a `LocalSamplingClient`) → `ml_log` → checkpoint if `step % save_every == 0` or preemption requested. Resume via `load_latest`. `num_batches` stops early (for smoke runs).
- Startup: `log_parity_warnings`; refuse `dataset_schedule`, `remove_constant_reward_groups`, `num_substeps` (not fields → chz rejects).

- [ ] **Step 1: Failing test** `test_two_steps_arithmetic_with_fake_vllm`: tiny model; `env="arithmetic"`; fake vLLM via `httpx.MockTransport` (pattern in `tests/test_multi_vllm.py`) returning random token ids + logprobs for completions and 200 for LoRA load; `num_batches=2`, `kl_penalty_coef=0.01`, `entropy_bonus_coef=0.01`, `use_rnd_curiosity=False`, `use_reasoning_rewards=False`, `save_every=1`. Assert: `metrics.jsonl` has 2 lines with keys `kl_policy_base`, `entropy/bonus_mean`, `loss`; a checkpoint dir for step 2 exists; the mock saw two `load_lora_adapter` calls. Second test `test_resume_continues_step_count`: run with `num_batches=1`, rerun same `log_path` with `num_batches=2` → second run performs exactly one step.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit.

### Task 8: SLURM launcher and docs

**Files:**
- Create: `slurm/sbatch_lean.sh` (zsh)
- Modify: `slurm/README.md`, `docs/pipeline.md` (new "Lean local trainer" section: how to launch, FSDP flag, GPU sizing table — Qwen3-8B on 48 GB replicated; gpt-oss-20b on 80 GB+ replicated or 48 GB with `fsdp=True`; documented Tinker mismatches)

**Interfaces:**
- `sbatch --nodes=N --gres=gpu:G --export=ALL,MODEL_NAME=...,VLLM_TP_SIZE=1,EXTRA_ARGS="fsdp=True ..." slurm/sbatch_lean.sh`: node 0 runs vLLM on GPUs `0..VLLM_TP_SIZE-1`; trainer ranks via `srun ... torchrun --nnodes=$SLURM_NNODES --nproc_per_node=<trainer GPUs on that node> --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m lean_rl.train backend fields...`; `VLLM_PLUGINS=lean_rl_gpt_oss_lm_head` when `MODEL_NAME` matches `gpt-oss` and `GPT_OSS_LM_HEAD=1`; SIGTERM forwarded to the trainer (which saves and exits) and then vLLM is stopped; `DRY_RUN=1` prints topology.

- [ ] **Step 1:** Implement; `zsh -n slurm/sbatch_lean.sh`; `DRY_RUN=1` with faked SLURM env for 1×8 and 2×8 GPUs prints the expected rank layout. Commit.
- [ ] **Step 2 (controller, GPU):** smoke runs with `num_batches=3`: Qwen3-8B 1 node replicated; gpt-oss-20b on 80 GB replicated; gpt-oss-20b on 8×48 GB with `fsdp=True`. Check `kl_policy_base≈0` at step 0, finite loss, adapter sync OK, resume after `scontrol requeue`.

### Task 9 (conditional — only if the Task 2 gate fails for gpt-oss experts): merged-weights serving

**Files:**
- Create: `lean_rl/merged_sync.py`; Modify: `lean_rl/loop.py`, `slurm/sbatch_lean.sh`
- Test: `tests/lean_rl/test_merged_sync.py`

**Interfaces:**
- Produces: `merge_and_export(trainer: Trainer, out_dir: str) -> str` (rank 0: merged bf16 HF checkpoint incl. tokenizer; gathers under FSDP) and `restart_vllm_with(model_dir: str, port: int, gpu_ids: str) -> str` (stop + `local_backend.vllm_proc.start_vllm_server` + `wait_until_ready`; returns URL). Loop flag `serve_merged: bool = False` (auto-True for gpt-oss when `LEAN_GPT_OSS_SERVE_MERGED=1`); when set, the LoRA hot-swap is replaced by merge+restart, and the sampling client binds to the base name (no adapter).

- [ ] **Step 1:** Failing test `test_merged_logprobs_equal_adapter_logprobs` (tiny model, CPU): merged model logprobs == PEFT model logprobs (atol 1e-5). Implement; PASS; commit.
- [ ] **Step 2 (controller, GPU):** gpt-oss-20b 3-step smoke with `serve_merged=True`.
