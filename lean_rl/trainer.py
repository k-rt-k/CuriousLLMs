"""Batched LoRA trainer matching Tinker's hyperparameters and token-sum loss.

Datum convention (cookbook ``assemble_training_data``): ``model_input`` holds the T input
tokens; ``loss_fn_inputs`` has ``target_tokens`` [T], ``logprobs`` [T] (sampler logprob of
each target), ``advantages`` [T] and ``mask`` [T] (1 on response tokens).

Memory: target logprobs are computed from the backbone's hidden states by running the
(PEFT-wrapped) ``lm_head`` + log-softmax-gather over chunks of ``logprob_chunk_tokens`` real
tokens, each chunk under activation checkpointing, so the full [B, T, V] logits never exist;
the decoder layers use HF gradient checkpointing (``gradient_checkpointing=True``).

Data is packed in order into right-padded micro-batches of at most ``micro_batch_tokens``
padded tokens (always at least one datum). The loss is the SUM over micro-batches of
``token_sum_loss`` -- no normalisation anywhere, as in Tinker.

Multi-rank (``dist`` from ``lean_rl.dist.init_dist``), SPMD: every rank calls ``train_step`` /
``logprobs`` with the same full data list and works on its token-balanced shard
(``lean_rl.dist.shard``). Gradients are SUMMED over ranks, never averaged, so the update
equals the single-process token-sum update.

- Replicated (default): full model per rank; rank 0's initial LoRA is broadcast at init; after
  backward the LoRA grads are SUM-all-reduced and every rank takes the identical AdamW step.
- ``fsdp=True``: FSDP2 ``fully_shard`` per decoder layer + a root over the remaining modules,
  ``MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=fp32)``, LoRA params kept fp32 and
  sharded, gradient divide factor 1 with SUM-only reductions. Every rank runs the same number
  of forwards/backwards (ranks with fewer or no micro-batches run zero-loss 1-token dummies).

Multi-rank contract -- EVERY rank must call these, in the same order, with the same args:
``train_step`` / ``_forward_backward``, ``logprobs`` (result on rank 0, None elsewhere),
``save_adapter`` (only rank 0 writes; FSDP gathers the LoRA shards), ``snapshot_init_adapter``
and ``state_dict_for_ckpt`` (FSDP gathers; every rank gets the full dict, save it on rank 0),
``load_adapter`` / ``load_ckpt_state`` (every rank reads the same file / dict; no
communication, but skipping a rank desyncs replicas or leaves its shards stale).
``train_step(..., should_stop=)`` adds one flag agreement per micro-batch slot: pass
``should_stop`` on every rank or on none; it returns None on every rank if any rank stopped.
In replicated mode save_adapter / snapshot / state_dict_for_ckpt happen to be local, but call
them on all ranks anyway so the code is mode-independent. ``as_ref_client`` computes on the
calling rank alone: fine in replicated mode; under FSDP every rank must make the same calls.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, Sequence

import tinker
import torch
import torch.distributed as tdist
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils.checkpoint import checkpoint
from peft import (
    PeftConfig,
    get_peft_model,
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
from transformers import AutoConfig, AutoModelForCausalLM

from lean_rl import dist as ldist
from lean_rl.losses import token_sum_loss
from lean_rl.lora import log_parity_warnings, lora_placement, make_lora_config

logger = logging.getLogger(__name__)

Which = Literal["policy", "base", "init"]

# Replicated grad all-reduce bucket size (elements).
_BUCKET_NUMEL = 1 << 26
# Dummy micro-batch for ranks that must join FSDP collectives without data of their own.
_DUMMY_PAIR: tuple[list[int], list[int]] = ([0], [0])


def _td(datum: tinker.Datum, key: str, dtype: torch.dtype) -> torch.Tensor:
    return datum.loss_fn_inputs[key].to_torch().to(dtype)


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def _is_mxfp4(hf_config: Any) -> bool:
    """True for a pre-quantized MXFP4 checkpoint (``quantization_config.quant_method``)."""
    q = getattr(hf_config, "quantization_config", None)
    method = q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", None)
    return getattr(method, "value", method) == "mxfp4"


def _gpt_oss_quant_kw() -> dict[str, Any]:
    """``from_pretrained`` kwargs that load gpt-oss MXFP4 experts as plain (dequantized) weights."""
    from transformers import Mxfp4Config

    return {"quantization_config": Mxfp4Config(dequantize=True)}


def _pack(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Group indices in order so that len(group) * max_len(group) <= budget (min one item)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i, n in enumerate(lengths):
        new_max = max(cur_max, n)
        if cur and (len(cur) + 1) * new_max > budget:
            groups.append(cur)
            cur, new_max = [], n
        cur.append(i)
        cur_max = new_max
    if cur:
        groups.append(cur)
    return groups


class RefLogprobClient:
    """Duck-types ``tinker.SamplingClient.compute_logprobs_async`` for cookbook KL code."""

    def __init__(self, trainer: "Trainer", which: Which):
        self._trainer = trainer
        self._which = which

    async def compute_logprobs_async(self, model_input: tinker.ModelInput) -> list[float | None]:
        tokens = model_input.to_ints()
        if len(tokens) < 2:
            return [None] * len(tokens)
        (lp,) = self._trainer._logprobs_for([(tokens[:-1], tokens[1:])], self._which)
        return [None] + lp.tolist()


class _LogpHead(nn.Module):
    """Decoder + chunked lm_head target logprobs; the FSDP root module when ``fsdp=True``.

    Running everything through one module call lets FSDP's root hooks gather the
    embedding/norm/lm_head(+LoRA) shards once per forward; the per-chunk ``lm_head`` calls
    (and their checkpoint recomputes) then use the already-unsharded root params.
    """

    def __init__(self, model: nn.Module, decoder: nn.Module, lm_head: nn.Module, chunk: int):
        super().__init__()
        self.model = model
        # Plain attributes (not submodules): they are already registered under self.model.
        self.__dict__["_decoder"] = decoder
        self.__dict__["_lm_head"] = lm_head
        self.chunk = chunk

    def _chunk_logp(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self._lm_head(h).float()
        return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    def forward(self, ids: torch.Tensor, tgt: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """fp32 logprobs [N] of the targets at the N real (attention-mask) positions."""
        hidden = self._decoder(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        real = attn.bool()
        h, t = hidden[real], tgt[real]  # [N, H], [N]: only real (unpadded) positions
        grad = torch.is_grad_enabled()
        parts = []
        for s in range(0, h.shape[0], self.chunk):
            hc, tc = h[s : s + self.chunk], t[s : s + self.chunk]
            if grad:
                # Only hc is saved; the chunk's [n, V] logits are recomputed in backward.
                parts.append(checkpoint(self._chunk_logp, hc, tc, use_reentrant=False))
            else:
                parts.append(self._chunk_logp(hc, tc))
        return torch.cat(parts)


class Trainer:
    def __init__(
        self,
        base_model: str,
        *,
        rank: int,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        train_unembed: bool = True,
        gpt_oss_lm_head: bool = False,
        dist: ldist.DistInfo | None = None,
        fsdp: bool = False,
        micro_batch_tokens: int = 16384,
        logprob_chunk_tokens: int = 2048,
        gradient_checkpointing: bool = True,
        attn_implementation: str | None = None,
    ):
        if dist is not None and dist.world_size <= 1:
            dist = None
        if fsdp and dist is None:
            logger.warning("fsdp=True ignored: single rank (no dist)")
            fsdp = False
        if dist is not None and not tdist.is_initialized():
            raise RuntimeError("dist given but torch.distributed is not initialised (init_dist)")
        self.base_model = base_model
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.dist = dist
        self.fsdp = fsdp
        self.micro_batch_tokens = micro_batch_tokens
        self.logprob_chunk_tokens = logprob_chunk_tokens

        hf_config = AutoConfig.from_pretrained(base_model)
        is_gpt_oss = getattr(hf_config, "model_type", "") == "gpt_oss"
        if attn_implementation is None:
            # HF gpt-oss needs eager attention for its attention sinks.
            attn_implementation = "eager" if is_gpt_oss else "sdpa"
        # Weight staging (bounded host RAM):
        # - FSDP: only rank 0 materialises the checkpoint (on CPU); other ranks build on the
        #   meta device. After fully_shard, rank 0's full state (base weights AND its LoRA
        #   init) is broadcast and each rank keeps its shard (_load_fsdp_from_rank0).
        # - Replicated / single on CUDA: load straight onto the GPU (no CPU copy).
        full_sd: dict[str, torch.Tensor] | None = None
        load_kw: dict[str, Any] = dict(torch_dtype=dtype, attn_implementation=attn_implementation)
        # from_pretrained only: the gpt-oss checkpoints are MXFP4. Without an explicit dequantize,
        # transformers keeps MXFP4 when triton>=3.4 + kernels are installed (experts become
        # *_blocks / *_scales), which breaks the PEFT target_parameters on the experts and the
        # plain (from_config) parameter layout FSDP ranks > 0 build on meta. Only for
        # pre-quantized checkpoints: on a plain one the config would quantize on the fly.
        dequant = is_gpt_oss and _is_mxfp4(hf_config)
        pretrained_kw = dict(load_kw, **_gpt_oss_quant_kw() if dequant else {})
        if fsdp and not dist.is_main:
            with torch.device("meta"):
                base = AutoModelForCausalLM.from_config(hf_config, **load_kw)
        elif fsdp:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True, **pretrained_kw
            )
        elif torch.device(device).type == "cuda":
            # MXFP4 dequantize runs on the *current* CUDA device regardless of device_map.
            if torch.device(device).index is not None:
                torch.cuda.set_device(device)
            base = AutoModelForCausalLM.from_pretrained(
                base_model, device_map={"": device}, **pretrained_kw
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(base_model, **pretrained_kw).to(device)
        for p in base.parameters():
            p.requires_grad = False
        if getattr(base.config, "final_logit_softcapping", None):
            raise NotImplementedError("final_logit_softcapping is not supported by chunked logprobs")
        if gradient_checkpointing and base.supports_gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        placement = lora_placement(
            base, train_unembed=train_unembed, gpt_oss_lm_head=gpt_oss_lm_head
        )
        model_type = getattr(base.config, "model_type", "")
        log_parity_warnings(placement, model_type)
        self.placement = placement
        self.model = get_peft_model(base, make_lora_config(placement, rank))
        # Dropout is 0, so train/eval only toggles HF gradient checkpointing; train mode is
        # entered just for the grad-enabled forward in _forward_backward.
        self.model.eval()
        inner = self.model.get_base_model()
        self._decoder = inner.get_decoder()
        self._lm_head = inner.get_output_embeddings()  # PEFT LoRA wrapper when adapted
        self._head = _LogpHead(self.model, self._decoder, self._lm_head, logprob_chunk_tokens)

        self._fsdp_modules: list[Any] = []
        if fsdp:
            if dist.is_main:
                full_sd = self._head.state_dict()  # references to rank 0's CPU tensors
            self._apply_fsdp()
            self._load_fsdp_from_rank0(full_sd)
            del full_sd
            self._fsdp_lazy_init()
        elif dist is not None:
            self._broadcast_lora_from_rank0()

        self.optimizer: torch.optim.AdamW | None = None
        self._init_snapshot: dict[str, torch.Tensor] | None = None
        self.step = 0

    # ------------------------------------------------------------------ multi-rank
    @property
    def is_main(self) -> bool:
        return self.dist is None or self.dist.is_main

    @torch.no_grad()
    def _broadcast_lora_from_rank0(self) -> None:
        """Replicas/shards start from rank 0's LoRA init whatever each rank's seed."""
        for _, p in self._named_lora():
            buf = p.data.to(self.device)
            tdist.broadcast(buf, src=0)
            p.data.copy_(buf)

    def _apply_fsdp(self) -> None:
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

        mesh = DeviceMesh.from_group(tdist.group.WORLD, device_type=torch.device(self.device).type)
        mp = MixedPrecisionPolicy(param_dtype=self.dtype, reduce_dtype=torch.float32)
        for layer in self._decoder.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
        fully_shard(self._head, mesh=mesh, mp_policy=mp)
        self._fsdp_modules = [m for m in self._head.modules() if isinstance(m, FSDPModule)]
        for m in self._fsdp_modules:
            # Token-sum semantics: reduce-scatter is a plain SUM (no 1/world_size, no AVG op).
            m.set_gradient_divide_factor(1.0)
            m.set_force_sum_reduction_for_comms(True)

    @torch.no_grad()
    def _load_fsdp_from_rank0(self, full_sd: dict[str, torch.Tensor] | None) -> None:
        """Materialise meta shards (ranks > 0) and fill every shard from rank 0's full state."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        dev = torch.device(self.device)
        if not self.is_main:
            self._head.to_empty(device=dev)
        set_model_state_dict(
            self._head, full_sd if self.is_main else {},
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        # Non-persistent buffers (e.g. rotary inv_freq) are not in the state dict.
        for b in self._head.buffers():
            tdist.broadcast(b, src=0)

    @torch.no_grad()
    def _fsdp_lazy_init(self) -> None:
        """Run one tiny forward with the adapter enabled so FSDP2 lazy-inits now.

        FSDP2 fixes each param group's gradient dtypes (orig/reduce) at its first forward, from
        the params that have ``requires_grad`` at that moment. PEFT's ``disable_adapter()`` sets
        ``requires_grad=False`` on every LoRA param, so if the first forward were a KL
        ``which="base"`` pass the groups would see no trainable params: LoRA grads would then be
        reduced in bf16 and assigned to the fp32 shards (torch >= 2.11 raises on that). Every
        rank calls this from ``__init__``, so the all-gathers line up.
        """
        self._forward_logp([_DUMMY_PAIR])
        self._reshard()

    def _reshard(self) -> None:
        """Free unsharded FSDP params (the root stays unsharded after a no-grad forward).

        Must precede any in-place write to the sharded LoRA params, else the next forward
        would reuse the stale unsharded copy.
        """
        for m in self._fsdp_modules:
            m.reshard()

    def _sync_max(self, n: int, *, force: bool = False) -> int:
        """Max of ``n`` over ranks: under FSDP (every rank must run as many forwards), or in
        replicated mode with ``force`` (every rank must make as many preemption checks)."""
        if not self.fsdp and not (force and self.dist is not None):
            return n
        t = torch.tensor([n], dtype=torch.int64, device=self.device)
        tdist.all_reduce(t, op=tdist.ReduceOp.MAX)
        return int(t.item())

    def _all_reduce_sum(self, values: list[float]) -> list[float]:
        if self.dist is None:
            return values
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        tdist.all_reduce(t, op=tdist.ReduceOp.SUM)
        return t.tolist()

    @torch.no_grad()
    def _all_reduce_lora_grads(self) -> None:
        """Replicated mode: SUM-all-reduce LoRA grads (a param keeps grad None on every rank
        only if no rank produced one, matching the single-process AdamW skip)."""
        params = self._trainable()
        has = torch.tensor([float(p.grad is not None) for p in params], device=self.device)
        tdist.all_reduce(has, op=tdist.ReduceOp.MAX)
        live = [p for p, h in zip(params, has.tolist()) if h > 0]
        for p in live:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        bucket: list[torch.nn.Parameter] = []
        numel = 0
        for p in live + [None]:  # type: ignore[list-item]
            if p is not None and (not bucket or numel + p.numel() <= _BUCKET_NUMEL):
                bucket.append(p)
                numel += p.numel()
                continue
            if bucket:
                flat = torch.cat([q.grad.reshape(-1).float() for q in bucket])
                tdist.all_reduce(flat, op=tdist.ReduceOp.SUM)
                off = 0
                for q in bucket:
                    q.grad.copy_(flat[off : off + q.numel()].view_as(q.grad))
                    off += q.numel()
            bucket, numel = ([p], p.numel()) if p is not None else ([], 0)

    # ------------------------------------------------------------------ params
    def _named_lora(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _trainable(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self._named_lora()]

    def _lora_copy(self) -> dict[str, torch.Tensor]:
        """Local clones (DTensor shards under FSDP); for in-memory swaps only."""
        return {n: p.detach().clone() for n, p in self._named_lora()}

    def _lora_full(self) -> dict[str, torch.Tensor]:
        """Full fp CPU copies of the LoRA params (collective under FSDP)."""
        return {n: _full(p.detach()).to("cpu", copy=True) for n, p in self._named_lora()}

    @torch.no_grad()
    def _lora_assign(self, tensors: dict[str, torch.Tensor]) -> None:
        """Copy full tensors (or same-layout local clones) into the live LoRA params."""
        named = dict(self._named_lora())
        if set(named) != set(tensors):
            raise KeyError("LoRA parameter names do not match")
        self._reshard()
        for n, p in named.items():
            t = tensors[n]
            if isinstance(p, DTensor) and not isinstance(t, DTensor):
                # Every rank holds the full tensor: take the local shard, no communication.
                t = distribute_tensor(
                    t.to(device=p.device, dtype=p.dtype), p.device_mesh, p.placements,
                    src_data_rank=None,
                )
                p.copy_(t)
            else:
                p.copy_(t.to(device=p.device, dtype=p.dtype))

    # ------------------------------------------------------------------ forward
    def _forward_logp(self, pairs: Sequence[tuple[list[int], list[int]]]) -> torch.Tensor:
        """Right-padded [B, T] fp32 logprobs of targets given inputs (garbage on padding)."""
        T = max(len(x) for x, _ in pairs)
        B = len(pairs)
        ids = torch.zeros(B, T, dtype=torch.long)
        tgt = torch.zeros(B, T, dtype=torch.long)
        attn = torch.zeros(B, T, dtype=torch.long)
        for b, (x, y) in enumerate(pairs):
            assert len(x) == len(y), f"input len {len(x)} != target len {len(y)}"
            ids[b, : len(x)] = torch.tensor(x)
            tgt[b, : len(y)] = torch.tensor(y)
            attn[b, : len(x)] = 1
        ids, tgt, attn = ids.to(self.device), tgt.to(self.device), attn.to(self.device)
        logp = self._head(ids, tgt, attn)
        return logp.new_zeros(B, T).masked_scatter(attn.bool(), logp)

    @contextmanager
    def _adapter_mode(self, which: Which) -> Iterator[None]:
        if which == "policy":
            yield
        elif which == "base":
            with self.model.disable_adapter():
                yield
        elif which == "init":
            if self._init_snapshot is None:
                raise RuntimeError("snapshot_init_adapter() has not been called")
            current = self._lora_copy()
            self._lora_assign(self._init_snapshot)
            try:
                yield
            finally:
                self._lora_assign(current)
        else:
            raise ValueError(f"Unknown which={which!r}")

    def _logprobs_for(
        self, pairs: Sequence[tuple[list[int], list[int]]], which: Which, *, sync: bool = False
    ) -> list[torch.Tensor]:
        """Local logprobs. ``sync`` (FSDP): pad with dummy forwards to the max group count."""
        out: list[torch.Tensor | None] = [None] * len(pairs)
        groups = _pack([len(x) for x, _ in pairs], self.micro_batch_tokens) if pairs else []
        n_calls = self._sync_max(len(groups)) if sync else len(groups)
        with torch.no_grad(), self._adapter_mode(which):
            try:
                for k in range(n_calls):
                    if k >= len(groups):
                        self._forward_logp([_DUMMY_PAIR])
                        continue
                    group = groups[k]
                    logp = self._forward_logp([pairs[i] for i in group])
                    for b, i in enumerate(group):
                        out[i] = logp[b, : len(pairs[i][0])].float().cpu()
            finally:
                self._reshard()
        return out  # type: ignore[return-value]

    @staticmethod
    def _pair(datum: tinker.Datum) -> tuple[list[int], list[int]]:
        return datum.model_input.to_ints(), _td(datum, "target_tokens", torch.long).tolist()

    def logprobs(self, data: list[tinker.Datum], which: Which) -> list[torch.Tensor] | None:
        """Per-datum fp32 CPU logprobs of ``target_tokens`` under the chosen weights.

        Multi-rank: all ranks call with the same ``data``; each computes its shard and the
        results are gathered to rank 0 in input order. Non-main ranks return None.
        """
        pairs = [self._pair(d) for d in data]
        if self.dist is None:
            return self._logprobs_for(pairs, which)
        idx = ldist.shard_indices([len(x) for x, _ in pairs], self.dist)
        local = self._logprobs_for([pairs[i] for i in idx], which, sync=self.fsdp)
        parts = ldist.gather_object(list(zip(idx, local)))
        if parts is None:
            return None
        out: list[torch.Tensor | None] = [None] * len(pairs)
        for part in parts:
            for i, lp in part:
                out[i] = lp
        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]

    def as_ref_client(self, which: Which) -> RefLogprobClient:
        return RefLogprobClient(self, which)

    # ------------------------------------------------------------------ training
    def _forward_backward(
        self, data: list[tinker.Datum], loss_fn: str, should_stop: Callable[[], bool] | None = None
    ) -> dict[str, float] | None:
        """Accumulate the token-sum loss grads of ``data`` into ``.grad`` (no zeroing, no step).

        Multi-rank: all ranks pass the same full ``data``; each runs its shard and the grads
        end up SUMMED over ranks (all-reduce in replicated mode, FSDP reduce-scatter
        otherwise). Metrics are global sums on every rank.

        ``should_stop`` (preemption): polled after every micro-batch slot; the ranks agree on it
        (``lean_rl.dist.any_true``) the same number of times -- the micro-batch count is
        max-synced in both modes. If any rank says stop, every rank drops ALL ``.grad`` and
        returns None.
        """
        pairs_all = [self._pair(d) for d in data]
        idx = ldist.shard_indices([len(x) for x, _ in pairs_all], self.dist)
        local = [data[i] for i in idx]
        pairs = [pairs_all[i] for i in idx]
        replicated = self.dist is not None and not self.fsdp
        stash: list[torch.Tensor | None] = []
        if replicated:
            # Only this call's contribution is all-reduced; earlier (already reduced) grads
            # are added back afterwards.
            stash = [p.grad for p in self._trainable()]
            for p in self._trainable():
                p.grad = None

        loss_sum = n_resp = gap_sum = 0.0
        groups = _pack([len(x) for x, _ in pairs], self.micro_batch_tokens) if pairs else []
        n_calls = self._sync_max(len(groups), force=should_stop is not None)
        stopped = False
        self.model.train()  # activates HF gradient checkpointing; dropout is 0
        try:
            for k in range(n_calls):
                if k < len(groups):
                    stats = self._fwd_bwd_group(local, pairs, groups[k], loss_fn)
                    loss_sum += stats[0]
                    n_resp += stats[1]
                    gap_sum += stats[2]
                elif self.fsdp:
                    # Join FSDP's per-layer all-gathers / reduce-scatters with a zero loss.
                    (self._forward_logp([_DUMMY_PAIR]).sum() * 0.0).backward()
                if should_stop is not None and ldist.any_true(should_stop()):
                    stopped = True
                    break
        finally:
            self.model.eval()
        if stopped:
            for p in self._trainable():
                p.grad = None
            self._reshard()
            return None

        if replicated:
            self._all_reduce_lora_grads()
            with torch.no_grad():
                for p, g in zip(self._trainable(), stash):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad.add_(g)
        loss_sum, n_resp, gap_sum = self._all_reduce_sum([loss_sum, n_resp, gap_sum])
        return {
            "loss": loss_sum,
            "response_tokens": n_resp,
            "logprob_gap/sampler_vs_trainer_mean_abs": gap_sum / max(n_resp, 1.0),
        }

    def _fwd_bwd_group(self, data, pairs, group, loss_fn) -> tuple[float, float, float]:
        """Forward+backward one micro-batch; returns (loss sum, response tokens, |gap| sum)."""
        new_logp = self._forward_logp([pairs[i] for i in group])
        B, T = new_logp.shape
        samp = torch.zeros(B, T)
        adv = torch.zeros(B, T)
        mask = torch.zeros(B, T)
        for b, i in enumerate(group):
            n = len(pairs[i][0])
            samp[b, :n] = _td(data[i], "logprobs", torch.float32)
            adv[b, :n] = _td(data[i], "advantages", torch.float32)
            mask[b, :n] = _td(data[i], "mask", torch.float32)
        samp, adv, mask = (t.to(self.device) for t in (samp, adv, mask))
        loss = token_sum_loss(loss_fn, new_logp, samp, adv, mask)  # type: ignore[arg-type]
        loss.backward()
        with torch.no_grad():
            gap = (new_logp.detach() - samp).abs()
            gap_sum = torch.where(mask > 0, gap, torch.zeros_like(gap)).sum()
            return float(loss.item()), float(mask.sum().item()), float(gap_sum.item())

    def _ensure_optimizer(self) -> torch.optim.AdamW:
        if self.optimizer is None:
            # Hyperparameters are overwritten from tinker.AdamParams before every step.
            self.optimizer = torch.optim.AdamW(self._trainable(), lr=0.0, weight_decay=0.0)
        return self.optimizer

    def _optim_step(self, adam: tinker.AdamParams) -> float:
        """Clip (if requested) and step; returns the pre-clip global LoRA grad norm."""
        opt = self._ensure_optimizer()
        for g in opt.param_groups:
            g["lr"] = float(adam.learning_rate)
            g["betas"] = (float(adam.beta1), float(adam.beta2))
            g["eps"] = float(adam.eps)
            g["weight_decay"] = float(adam.weight_decay)
        # Params without a grad are skipped by AdamW, as in torch's default behaviour.
        params = [p for p in self._trainable() if p.grad is not None]
        clip = float(adam.grad_clip_norm)
        norm = 0.0
        if params:
            # FSDP: grads are DTensor shards; the norm is reduced across ranks by full_tensor.
            total = torch.nn.utils.get_total_norm([p.grad for p in params])
            total = _full(total)
            norm = float(total)
            if clip > 0:
                torch.nn.utils.clip_grads_with_norm_(params, clip, total)
        self._reshard()
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.step += 1
        return norm

    def train_step(
        self,
        data: list[tinker.Datum],
        loss_fn: str,
        adam: tinker.AdamParams,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, float] | None:
        """One optimizer step on ``data``. With ``should_stop`` (e.g. the preemption flag) the
        ranks agree on it after every micro-batch; if any rank says stop, the step is abandoned
        on every rank -- grads dropped, no optimizer step, weights / optimizer / ``step``
        untouched -- and None is returned (on every rank)."""
        if loss_fn not in ("ppo", "importance_sampling"):
            raise ValueError(f"Unsupported loss_fn: {loss_fn!r}")
        self._ensure_optimizer().zero_grad(set_to_none=True)
        metrics = self._forward_backward(data, loss_fn, should_stop)
        if metrics is None:
            return None
        metrics["grad_norm"] = self._optim_step(adam)
        return metrics

    # ------------------------------------------------------------------ adapters / state
    def save_adapter(self, path: str) -> None:
        """Write the PEFT adapter dir. Multi-rank: call on ALL ranks with the same path; only
        rank 0 writes (FSDP: the others take part in the LoRA gather). No barrier: only rank 0
        may read ``path`` afterwards unless the caller adds one (``lean_rl.dist.barrier``)."""
        # save_embedding_layers=False: with lm_head in target_modules PEFT's "auto" would also
        # write the full lm_head.base_layer weight, which vLLM rejects and which is huge.
        if self.fsdp:
            full = self._lora_full()
            if self.is_main:
                self.model.save_pretrained(path, state_dict=full, save_embedding_layers=False)
        elif self.is_main:
            self.model.save_pretrained(path, save_embedding_layers=False)

    def snapshot_init_adapter(self) -> None:
        """Freeze the current LoRA as the ``init`` reference (FSDP: collective, all ranks)."""
        self._init_snapshot = self._lora_full()

    def load_adapter(self, path: str) -> None:
        """Copy adapter weights from a PEFT dir into the live LoRA params; fresh optimizer.

        Multi-rank: call on ALL ranks with the same path (shared FS); each rank reads the file
        and keeps its replica / shard. Local (no collectives)."""
        cfg = self.model.peft_config["default"]
        saved = PeftConfig.from_pretrained(path)
        if int(saved.r) != int(cfg.r):
            raise ValueError(f"adapter rank {saved.r} != trainer rank {cfg.r}")
        weights = load_peft_weights(path, device="cpu")
        if self.fsdp:
            # PEFT's loader cannot write into DTensor params: map our param names to the keys
            # PEFT saves them under (via PEFT's own get_peft_model_state_dict on index tags)
            # and assign shards ourselves.
            names = [n for n, _ in self._named_lora()]
            tags = {n: torch.tensor(i) for i, n in enumerate(names)}
            saved_keys = get_peft_model_state_dict(
                self.model, state_dict=tags, adapter_name=self.model.active_adapter,
                save_embedding_layers=False,
            )
            key = {names[int(v)]: k for k, v in saved_keys.items()}
            missing = [n for n in names if n not in key or key[n] not in weights]
            unexpected = sorted(set(weights) - set(key.values()))
            if missing or unexpected:
                raise RuntimeError(f"adapter mismatch: missing={missing} unexpected={unexpected}")
            self._lora_assign({n: weights[key[n]] for n in names})
        else:
            before = self._lora_copy()
            res = set_peft_model_state_dict(self.model, weights)
            missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
            unexpected = list(getattr(res, "unexpected_keys", []))
            if missing or unexpected:
                self._lora_assign(before)
                raise RuntimeError(f"adapter mismatch: missing={missing} unexpected={unexpected}")
        self.optimizer = None

    def _optimizer_state_full(self) -> dict | None:
        if self.optimizer is None:
            return None
        sd = self.optimizer.state_dict()
        # Full (gathered) CPU tensors: the same format with or without FSDP. Copies, since
        # Optimizer.state_dict() returns references to the live moment tensors.
        state = {
            i: {k: (_full(v).to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                for k, v in s.items()}
            for i, s in sd["state"].items()
        }
        return {"state": state, "param_groups": copy.deepcopy(sd["param_groups"])}

    def state_dict_for_ckpt(self) -> dict:
        """Full CPU state (same format with or without FSDP, so checkpoints are portable).

        Multi-rank: call on ALL ranks (FSDP gathers shards); every rank gets the full dict,
        rank 0 saves it."""
        return {
            "lora": self._lora_full(),
            "optimizer": self._optimizer_state_full(),
            "init_snapshot": None if self._init_snapshot is None else {
                n: t.clone() for n, t in self._init_snapshot.items()
            },
            "step": self.step,
        }

    def load_ckpt_state(self, d: dict) -> None:
        """Inverse of ``state_dict_for_ckpt``. Multi-rank: call on ALL ranks with the same
        dict (each loads the file itself); local, each rank keeps its replica / shard."""
        self._lora_assign(d["lora"])
        self.optimizer = None
        if d.get("optimizer") is not None:
            sd = copy.deepcopy(d["optimizer"])
            if self.fsdp:
                params = self._trainable()  # optimizer state is indexed in this order
                for i, s in sd["state"].items():
                    p = params[int(i)]
                    for k, v in s.items():
                        if isinstance(p, DTensor) and isinstance(v, torch.Tensor) and v.shape == p.shape:
                            s[k] = distribute_tensor(
                                v.to(device=p.device), p.device_mesh, p.placements,
                                src_data_rank=None,
                            )
            self._ensure_optimizer().load_state_dict(sd)
        snap = d.get("init_snapshot")
        self._init_snapshot = None if snap is None else {n: t.clone() for n, t in snap.items()}
        self.step = int(d.get("step", 0))
