"""
LocalTrainingClient — duck-typed replacement for `tinker.TrainingClient`.

Backed by Hugging Face `transformers` + `peft` LoRA on one GPU, or data-parallel
across ranks when constructed with a `DistContext` (see local_backend/distributed.py:
rank 0 drives, other ranks serve commands, LoRA grads are SUM-all-reduced). Exposes:

  forward_backward_async(data, loss_fn) -> LocalFuture[ForwardBackwardOutput]
  optim_step_async(adam_params)          -> LocalFuture[OptimStepResponse]
  save_state_async(name)                 -> LocalFuture[SavedPath]
  save_weights_for_sampler_async(name)   -> LocalFuture[SavedPath]
  load_state_async(path)                 -> LocalFuture[None]
  create_sampling_client(path)           -> LocalSamplingClient
  get_tokenizer() -> transformers.PreTrainedTokenizer

After every successful `optim_step_async` the next `create_sampling_client(path)`
syncs the freshly-saved LoRA adapter into the running vLLM server so rollouts
use the up-to-date policy.

Datum shape consumed (matches tinker_cookbook.rl.data_processing.trajectory_to_data):
  Datum.model_input         : tinker.ModelInput   (length T, the INPUT tokens)
  Datum.loss_fn_inputs:
      target_tokens (int64)   : [T]  next-token targets
      logprobs      (float32) : [T]  sample-time logprob of target
      advantages    (float32) : [T]  per-token advantage (0 on prompt tokens)
      mask          (float32) : [T]  1 on response tokens, 0 on prompt
                                     (curiosity_train.remove_mask strips this
                                      before forward_backward, so we fall back
                                      to advantages != 0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional, Sequence

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import tinker
from tinker import TensorData
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_backend.distributed import Cmd, DistContext, send_command, shard_by_tokens
from local_backend.futures import LocalFuture, SavedPath
from local_backend.losses import compute_loss
from local_backend.weight_sync import sync_lora_to_vllm

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _td_to_tensor(td: TensorData) -> torch.Tensor:
    """tinker.TensorData -> torch.Tensor. Cheap, used everywhere."""
    return td.to_torch()


@dataclass
class _ForwardBackwardResult:
    """Quacks like tinker.ForwardBackwardOutput."""
    loss_fn_output_type: str
    loss_fn_outputs: List[dict]
    metrics: dict


@dataclass
class _OptimStepResponse:
    """Quacks like tinker.OptimStepResponse."""
    metrics: dict


# ----------------------------------------------------------------------
# LocalTrainingClient
# ----------------------------------------------------------------------


class LocalTrainingClient:
    def __init__(
        self,
        *,
        base_model: str,
        lora_rank: int,
        vllm_url: str,
        log_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        gpu_memory_fraction: float = 0.5,
        ppo_clip_eps: float = 0.2,
        lora_alpha: Optional[int] = None,
        lora_target_modules: str = "all-linear",
        lora_dropout: float = 0.0,
        dist: Optional[DistContext] = None,
    ):
        self.base_model = base_model
        # Data parallelism (see local_backend/distributed.py). With dist None or
        # world_size 1 every code path below is the single-process one.
        self.dist = dist
        self._dp = dist is not None and dist.world_size > 1
        if device is None:
            device = f"cuda:{dist.local_rank}" if (dist is not None and torch.cuda.is_available()) else "cuda:0"
        self.lora_rank = lora_rank
        self.vllm_url = vllm_url.rstrip("/")
        self.log_path = log_path or os.environ.get("LOG_DIR") or "."
        self.device = device
        self.dtype = dtype
        # Env override for models that don't support sdpa (e.g. GptOss → eager).
        env_attn = os.environ.get("LOCAL_ATTN_IMPL")
        self.attn_implementation = env_attn if env_attn else attn_implementation
        # Env-var overrides for sbatch wrappers that can't reach the chz CLI.
        env_mem = os.environ.get("LOCAL_TRAINER_GPU_MEM_FRAC")
        self.gpu_memory_fraction = float(env_mem) if env_mem else gpu_memory_fraction
        self.ppo_clip_eps = ppo_clip_eps
        # Tinker uses a fixed lora_alpha=32 regardless of rank (see
        # tinker_cookbook.hyperparam_utils.get_lora_lr_over_full_finetune_lr), so
        # the LoRA scale alpha/r — and hence the effective LR — matches Tinker.
        self.lora_alpha = lora_alpha if lora_alpha is not None else 32
        env_targets = os.environ.get("LOCAL_LORA_TARGETS")
        self.lora_target_modules = env_targets if env_targets else lora_target_modules
        # `target_modules` can be a comma-list (env var) or "all-linear" sentinel.
        if isinstance(self.lora_target_modules, str) and "," in self.lora_target_modules:
            self.lora_target_modules = [t.strip() for t in self.lora_target_modules.split(",") if t.strip()]
        self.lora_dropout = lora_dropout

        # Set later by _init_async
        self.model = None
        self.tokenizer = None
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.step_counter = 0
        self._current_adapter_name: Optional[str] = None
        self._adapter_swap_lock = asyncio.Lock()
        self._last_loss_value: Optional[float] = None
        self._adam_state: dict[str, float] = {
            "lr": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0,
        }

    # ------------------------------------------------------------------
    # Initialization (deferred — needs to be awaitable to match Tinker)
    # ------------------------------------------------------------------
    async def _init_async(self):
        from peft import LoraConfig, get_peft_model

        if torch.cuda.is_available() and self.gpu_memory_fraction < 1.0:
            try:
                dev = torch.device(self.device)
                dev_index = dev.index if dev.type == "cuda" and dev.index is not None else 0
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device=dev_index)
            except Exception as e:
                logger.warning("set_per_process_memory_fraction failed: %s", e)

        logger.info("Loading base model %s (dtype=%s, attn=%s)",
                    self.base_model, self.dtype, self.attn_implementation)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        # Freeze base; PEFT will mark LoRA params trainable.
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.train()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        logger.info("LoRA trainable parameters: %d", n_train)

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self._adam_state["lr"],
            betas=(self._adam_state["beta1"], self._adam_state["beta2"]),
            eps=self._adam_state["eps"],
            weight_decay=self._adam_state["weight_decay"],
        )
        self.optimizer.zero_grad(set_to_none=True)

        if self._dp:
            # Every rank must start from rank 0's LoRA init, whatever its RNG did.
            for p in self._trainable_params():
                torch_dist.broadcast(p.data, src=0)

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    def get_tokenizer(self):
        return self.tokenizer

    # ------------------------------------------------------------------
    # forward + backward
    # ------------------------------------------------------------------
    async def forward_backward_async(
        self,
        data: Sequence[tinker.Datum],
        loss_fn: Literal["ppo", "importance_sampling", "nll"] = "ppo",
    ) -> LocalFuture[_ForwardBackwardResult]:
        if self._dp:
            per_datum_logprobs, loss_sum, total_response_tokens = self._dp_forward_backward(
                data, loss_fn
            )
        else:
            per_datum_logprobs, loss_sum, total_response_tokens = self._local_forward_backward(
                data, loss_fn
            )

        denom = max(total_response_tokens, 1)
        mean_loss = loss_sum / denom
        self._last_loss_value = mean_loss
        result = _ForwardBackwardResult(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=[{"logprobs": td} for td in per_datum_logprobs],
            metrics={
                "loss": mean_loss,
                "response_tokens": float(total_response_tokens),
            },
        )
        return LocalFuture(result)

    def _local_forward_backward(
        self,
        data: Sequence[tinker.Datum],
        loss_fn: Literal["ppo", "importance_sampling", "nll"],
    ) -> tuple[list[TensorData], float, int]:
        """Forward+backward on this process's data; accumulates into .grad.

        Returns (per-datum logprobs, token-sum of the loss, # response tokens).
        The gradient accumulated is the token-SUM of the loss (per-datum mean
        reweighted by its response-token count), never a mean over data.
        """
        per_datum_logprobs: list[TensorData] = []
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_response_tokens = 0

        for datum in data:
            input_ids = torch.tensor(
                datum.model_input.to_ints(), dtype=torch.long, device=self.device
            ).unsqueeze(0)  # [1, T]
            T = input_ids.shape[1]

            target_tokens = _td_to_tensor(datum.loss_fn_inputs["target_tokens"]).to(
                device=self.device, dtype=torch.long
            )
            assert target_tokens.shape[0] == T, f"target_tokens {target_tokens.shape} vs T={T}"

            # SFT (NLL) reads only target_tokens + weights. RL reads logprobs+advantages+mask.
            sample_logp = None
            advantages = None
            mask = None
            weights = None
            if loss_fn == "nll":
                weights = _td_to_tensor(datum.loss_fn_inputs["weights"]).to(
                    device=self.device, dtype=torch.float32
                )
                assert weights.shape[0] == T
            else:
                sample_logp = _td_to_tensor(datum.loss_fn_inputs["logprobs"]).to(
                    device=self.device, dtype=torch.float32
                )
                advantages = _td_to_tensor(datum.loss_fn_inputs["advantages"]).to(
                    device=self.device, dtype=torch.float32
                )
                if "mask" in datum.loss_fn_inputs:
                    mask = _td_to_tensor(datum.loss_fn_inputs["mask"]).to(
                        device=self.device, dtype=torch.float32
                    )
                else:
                    # remove_mask in curiosity_train.py strips it; recover from advantages.
                    mask = (advantages != 0).to(torch.float32)
                assert advantages.shape[0] == T
                assert sample_logp.shape[0] == T
                assert mask.shape[0] == T

            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits.squeeze(0)  # [T, V]
            logprobs_full = F.log_softmax(logits.float(), dim=-1)  # [T, V]
            new_logp = logprobs_full.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)  # [T]

            loss = compute_loss(
                loss_fn,
                new_logp=new_logp,
                sample_logp=sample_logp,
                advantages=advantages,
                mask=mask,
                weights=weights,
                ppo_clip_eps=self.ppo_clip_eps,
            )
            if loss_fn == "nll":
                n_resp = int((weights > 0).sum().item())
            else:
                n_resp = int(mask.sum().item())
            # Reweight per-datum mean loss by # response tokens so the overall
            # gradient is the per-token average across the whole micro-batch.
            weight = max(n_resp, 1)
            (loss * weight).backward()
            total_loss = total_loss.detach() + (loss.detach() * weight)
            total_response_tokens += n_resp

            per_datum_logprobs.append(TensorData.from_torch(new_logp.detach().cpu()))

        return per_datum_logprobs, float(total_loss.item()), total_response_tokens

    # ------------------------------------------------------------------
    # Data parallelism (driver side + shared per-rank bodies)
    # ------------------------------------------------------------------
    def _run_command(self, cmd: Cmd, payloads: list[Any], fn):
        """Driver: broadcast `cmd`, then run the same body locally.

        `dist.busy` stays True if `fn` raises, so `shutdown_workers` knows the
        workers are stuck in a collective and must not be sent another command.
        """
        own = send_command(self.dist, cmd, payloads)
        self.dist.busy = True
        out = fn(own)
        self.dist.busy = False
        return out

    def _dp_forward_backward(
        self,
        data: Sequence[tinker.Datum],
        loss_fn: Literal["ppo", "importance_sampling", "nll"],
    ) -> tuple[list[TensorData], float, int]:
        data = list(data)
        world_size = self.dist.world_size
        lengths = [len(d.model_input.to_ints()) for d in data]
        shards = shard_by_tokens(lengths, world_size)
        payloads = [((idx, [data[i] for i in idx]), loss_fn) for idx in shards]
        gathered = self._run_command(
            Cmd.FWD_BWD, payloads, lambda p: self._do_fwd_bwd(p[0], p[1])
        )

        logprobs: list[Optional[TensorData]] = [None] * len(data)
        loss_sum = 0.0
        n_tokens = 0
        for indices, rank_logprobs, rank_loss_sum, rank_tokens in gathered:
            for i, lp in zip(indices, rank_logprobs):
                logprobs[i] = lp
            loss_sum += rank_loss_sum
            n_tokens += rank_tokens
        assert all(lp is not None for lp in logprobs), "missing logprobs after gather"
        return logprobs, loss_sum, n_tokens

    def _do_fwd_bwd(self, shard, loss_fn) -> Optional[list]:
        """Every rank: local fwd/bwd on `shard`, SUM-all-reduce grads, gather outputs.

        `shard` is `(datum_indices, datums)` (possibly empty). Returns the
        gathered `(indices, logprobs, loss_sum, n_tokens)` per rank on the
        driver, None on workers.
        """
        indices, datums = shard
        # Grads from earlier fwd_bwd calls since the last optim step are
        # already globally reduced; set them aside so only this call's fresh
        # contribution is all-reduced, then add them back.
        params = self._trainable_params()
        prev = [p.grad for p in params]
        for p in params:
            p.grad = None
        logprobs, loss_sum, n_tokens = self._local_forward_backward(datums, loss_fn)
        self._allreduce_lora_grads()
        for p, g in zip(params, prev):
            if g is not None:
                p.grad.add_(g)

        gathered = [None] * self.dist.world_size if self.dist.is_driver else None
        torch_dist.gather_object(
            (list(indices), logprobs, loss_sum, n_tokens),
            gathered,
            dst=0,
            group=self.dist.obj_group,
        )
        return gathered

    def _allreduce_lora_grads(self) -> None:
        """SUM-all-reduce every trainable grad (zeros where None) in flat buffers."""
        grads_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for p in self._trainable_params():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
        for grads in grads_by_dtype.values():
            flat = torch._utils._flatten_dense_tensors(grads)
            torch_dist.all_reduce(flat, op=torch_dist.ReduceOp.SUM)
            for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                g.copy_(synced)

    def _maybe_check_sync(self) -> None:
        """LOCAL_DIST_CHECK_SYNC=1: every 50 steps, assert LoRA params agree across ranks."""
        if not self._dp or os.environ.get("LOCAL_DIST_CHECK_SYNC") != "1":
            return
        if self.step_counter % 50 != 0:
            return
        with torch.no_grad():
            total = torch.stack(
                [p.detach().to(torch.float64).sum() for p in self._trainable_params()]
            ).sum()
        hi, lo = total.clone(), total.clone()
        torch_dist.all_reduce(hi, op=torch_dist.ReduceOp.MAX)
        torch_dist.all_reduce(lo, op=torch_dist.ReduceOp.MIN)
        if hi.item() != lo.item():
            raise RuntimeError(
                f"LoRA params diverged across ranks at step {self.step_counter}: "
                f"param-sum max {hi.item()!r} != min {lo.item()!r}"
            )

    # ------------------------------------------------------------------
    # optim_step
    # ------------------------------------------------------------------
    async def optim_step_async(self, adam_params: tinker.AdamParams) -> LocalFuture[_OptimStepResponse]:
        if self._dp:
            metrics = self._run_command(
                Cmd.OPTIM, [adam_params] * self.dist.world_size, self._do_optim
            )
        else:
            metrics = self._do_optim(adam_params)
        return LocalFuture(_OptimStepResponse(metrics=metrics))

    def _do_optim(self, adam_params: tinker.AdamParams) -> dict:
        """Every rank: identical AdamW step on the (already all-reduced) grads."""
        self._set_adam(adam_params)
        # Optional gradient clipping (after the all-reduce, so identical on every rank)
        grad_clip = float(getattr(adam_params, "grad_clip_norm", 0.0) or 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_counter += 1
        self._maybe_check_sync()
        metrics = {"step": float(self.step_counter)}
        if self._last_loss_value is not None:
            metrics["prev_loss"] = self._last_loss_value
        return metrics

    def _set_adam(self, adam_params: tinker.AdamParams):
        for group in self.optimizer.param_groups:
            group["lr"] = float(adam_params.learning_rate)
            group["betas"] = (float(adam_params.beta1), float(adam_params.beta2))
            group["eps"] = float(adam_params.eps)
            group["weight_decay"] = float(getattr(adam_params, "weight_decay", 0.0) or 0.0)
        self._adam_state.update(
            lr=float(adam_params.learning_rate),
            beta1=float(adam_params.beta1),
            beta2=float(adam_params.beta2),
            eps=float(adam_params.eps),
            weight_decay=float(getattr(adam_params, "weight_decay", 0.0) or 0.0),
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _state_dir(self, name: str) -> str:
        d = os.path.join(self.log_path, "states", name)
        os.makedirs(d, exist_ok=True)
        return d

    def _adapter_dir(self, name: str) -> str:
        d = os.path.join(self.log_path, "adapters", name)
        os.makedirs(d, exist_ok=True)
        return d

    async def save_state_async(self, name: str) -> LocalFuture[SavedPath]:
        d = self._state_dir(name)
        # Persist adapter alongside optimizer for a fully-resumable checkpoint.
        self.model.save_pretrained(d)
        torch.save(self.optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
        meta = {
            "step": self.step_counter,
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "adam": self._adam_state,
        }
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        try:
            torch.save(
                {
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                os.path.join(d, "rng.pt"),
            )
        except Exception as e:
            logger.warning("RNG save failed: %s", e)
        logger.info("save_state(%s) -> %s", name, d)
        return LocalFuture(SavedPath(path=d))

    async def save_weights_for_sampler_async(self, name: str) -> LocalFuture[SavedPath]:
        d = self._adapter_dir(name)
        self.model.save_pretrained(d)
        logger.info("save_weights_for_sampler(%s) -> %s", name, d)
        return LocalFuture(SavedPath(path=d))

    async def save_weights_and_get_sampling_client_async(self, name: str):
        sampler_fut = await self.save_weights_for_sampler_async(name)
        sampler_path = (await sampler_fut.result_async()).path
        return self.create_sampling_client(sampler_path)

    async def load_state_async(self, path: str) -> LocalFuture[None]:
        # Accepts either a state_dir (with optimizer.pt) or a bare adapter dir.
        # Under data parallelism every rank loads the same path (it must be on
        # storage all ranks can see).
        if self._dp:
            self._check_adapter_dir(path)
            self._run_command(Cmd.LOAD_STATE, [path] * self.dist.world_size, self._do_load_state)
        else:
            self._do_load_state(path)
        return LocalFuture(None)

    @staticmethod
    def _check_adapter_dir(path: str) -> str:
        cfg = os.path.join(path, "adapter_config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"No adapter_config.json found under {path}")
        return cfg

    def _do_load_state(self, path: str) -> None:
        self._check_adapter_dir(path)

        # PEFT's load_adapter STACKS rather than replaces when adapter_name
        # already exists. Delete the in-memory adapter first.
        try:
            existing = set(getattr(self.model, "peft_config", {}).keys())
        except Exception:
            existing = set()
        if "default" in existing:
            try:
                self.model.delete_adapter("default")
            except Exception as e:
                logger.warning("delete_adapter('default') failed before reload: %s", e)
        self.model.load_adapter(path, adapter_name="default", is_trainable=True)
        try:
            self.model.set_adapter("default")
        except Exception:
            pass

        opt_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(opt_path):
            # weights_only=True is safe here: optimizer state is tensors+ints only.
            state = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state)

        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.step_counter = int(meta.get("step", self.step_counter))
            if "adam" in meta:
                self._adam_state.update(meta["adam"])

        rng_path = os.path.join(path, "rng.pt")
        if os.path.exists(rng_path):
            try:
                # RNG file holds ByteTensors and a tuple — not safe with weights_only.
                rng = torch.load(rng_path, map_location="cpu", weights_only=False)
                if rng.get("torch_rng") is not None:
                    torch.set_rng_state(rng["torch_rng"])
                if rng.get("cuda_rng_all") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda_rng_all"])
            except Exception as e:
                logger.warning("RNG restore failed: %s", e)

        logger.info("load_state(%s) ok (step=%d)", path, self.step_counter)

    async def load_adapter_weights_async(self, path: str) -> LocalFuture[None]:
        """Warm-start: load only adapter weights. Leaves optimizer/step_counter/RNG fresh.

        Use this to bootstrap a new RL run from a previously-trained LoRA (e.g.
        a cold-start SFT adapter). Compare to `load_state_async`, which restores
        the full training state (optimizer momentum, step counter, RNG).
        Accepts either a bare adapter dir or a state dir (both contain
        adapter_config.json).
        """
        # Serialize against concurrent vLLM weight-syncs so a sampler creation
        # can't race with the in-flight adapter swap.
        async with self._adapter_swap_lock:
            if self._dp:
                self._validate_adapter(path)  # fail on the driver before involving workers
                self._run_command(
                    Cmd.LOAD_ADAPTER, [path] * self.dist.world_size, self._do_load_adapter
                )
            else:
                self._do_load_adapter(path)
        return LocalFuture(None)

    def _validate_adapter(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError(
                "LocalTrainingClient._init_async() must run before load_adapter_weights_async; "
                "use LocalServiceClient.create_lora_training_client_async to construct."
            )
        cfg = self._check_adapter_dir(path)

        # Cheap up-front rank check — PEFT will silently load a mismatched-rank
        # adapter, leading to wrong tensor shapes later. Fail loudly here.
        try:
            with open(cfg) as f:
                adapter_cfg = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {cfg}: {e}") from e
        cfg_rank = adapter_cfg.get("r")
        if cfg_rank is not None and int(cfg_rank) != int(self.lora_rank):
            raise ValueError(
                f"LoRA rank mismatch loading {path}: adapter r={cfg_rank}, "
                f"trainer lora_rank={self.lora_rank}. Re-init the trainer with the same rank."
            )

    def _do_load_adapter(self, path: str) -> None:
        """Every rank: replace the LoRA adapter with `path`, fresh optimizer, step 0."""
        self._validate_adapter(path)
        try:
            existing = set(getattr(self.model, "peft_config", {}).keys())
        except Exception:
            existing = set()
        if "default" in existing:
            try:
                self.model.delete_adapter("default")
            except Exception as e:
                logger.warning("delete_adapter('default') failed before warm-start: %s", e)
        self.model.load_adapter(path, adapter_name="default", is_trainable=True)
        # set_adapter MUST succeed — silent failure leaves the wrong adapter
        # active and training proceeds with wrong weights. Let real errors propagate.
        self.model.set_adapter("default")

        # Rebuild optimizer over the freshly-loaded LoRA params so AdamW state
        # is fresh. (The previous optimizer was bound to deleted param tensors.)
        # _adam_state retains lr/betas/eps/weight_decay (set by the user via
        # optim_step or by _init_async defaults); only the momentum state resets.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                f"No trainable LoRA params after load_adapter({path}). Adapter may be corrupted."
            )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self._adam_state["lr"],
            betas=(self._adam_state["beta1"], self._adam_state["beta2"]),
            eps=self._adam_state["eps"],
            weight_decay=self._adam_state["weight_decay"],
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.step_counter = 0
        self._last_loss_value = None

        logger.info("load_adapter_weights(%s) ok — fresh optimizer, step=0", path)

    # ------------------------------------------------------------------
    # SamplingClient creation (with weight sync)
    # ------------------------------------------------------------------
    def create_sampling_client(self, sampler_path: Optional[str] = None, **_unused):
        # Synchronous variant: schedule the async one on a fresh loop if needed.
        from local_backend.sampling_client import LocalSamplingClient
        from local_backend.service_client import _adapter_name_from_path

        if sampler_path is None:
            return LocalSamplingClient(
                vllm_url=self.vllm_url,
                base_model_name=self.base_model,
                adapter_name=None,
                tokenizer=self.tokenizer,
            )

        adapter_name = _adapter_name_from_path(sampler_path)
        # Sync call: ensure no concurrent swap is in flight via a sync mutex.
        # Use the asyncio.Lock by running it inside the running loop if any.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            # Caller should have awaited create_sampling_client_async; fall back
            # to best-effort no-lock here (still safe if not contended).
            sync_lora_to_vllm(
                adapter_dir=sampler_path,
                name=adapter_name,
                vllm_url=self.vllm_url,
                prev_name=self._current_adapter_name,
            )
            self._current_adapter_name = adapter_name
        else:
            sync_lora_to_vllm(
                adapter_dir=sampler_path,
                name=adapter_name,
                vllm_url=self.vllm_url,
                prev_name=self._current_adapter_name,
            )
            self._current_adapter_name = adapter_name

        return LocalSamplingClient(
            vllm_url=self.vllm_url,
            base_model_name=self.base_model,
            adapter_name=adapter_name,
            tokenizer=self.tokenizer,
        )

    async def create_sampling_client_async(self, sampler_path: Optional[str] = None, **_unused):
        from local_backend.sampling_client import LocalSamplingClient
        from local_backend.service_client import _adapter_name_from_path

        if sampler_path is None:
            return LocalSamplingClient(
                vllm_url=self.vllm_url,
                base_model_name=self.base_model,
                adapter_name=None,
                tokenizer=self.tokenizer,
            )

        adapter_name = _adapter_name_from_path(sampler_path)
        async with self._adapter_swap_lock:
            await asyncio.to_thread(
                sync_lora_to_vllm,
                sampler_path,
                adapter_name,
                self.vllm_url,
                self._current_adapter_name,
            )
            self._current_adapter_name = adapter_name
        return LocalSamplingClient(
            vllm_url=self.vllm_url,
            base_model_name=self.base_model,
            adapter_name=adapter_name,
            tokenizer=self.tokenizer,
        )
