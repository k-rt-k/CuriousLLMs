"""Single-process batched LoRA trainer matching Tinker's hyperparameters and token-sum loss.

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
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Sequence

import tinker
import torch
from torch.utils.checkpoint import checkpoint
from peft import PeftConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM

from lean_rl.losses import token_sum_loss
from lean_rl.lora import log_parity_warnings, lora_placement, make_lora_config

logger = logging.getLogger(__name__)

Which = Literal["policy", "base", "init"]


def _td(datum: tinker.Datum, key: str, dtype: torch.dtype) -> torch.Tensor:
    return datum.loss_fn_inputs[key].to_torch().to(dtype)


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
        dist: Any | None = None,
        fsdp: bool = False,
        micro_batch_tokens: int = 16384,
        logprob_chunk_tokens: int = 2048,
        gradient_checkpointing: bool = True,
    ):
        if dist is not None or fsdp:
            raise NotImplementedError("multi-rank DP/FSDP is not implemented yet (Task 5)")
        self.base_model = base_model
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.micro_batch_tokens = micro_batch_tokens
        self.logprob_chunk_tokens = logprob_chunk_tokens

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
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

        self.optimizer: torch.optim.AdamW | None = None
        self._init_snapshot: dict[str, torch.Tensor] | None = None
        self.step = 0

    # ------------------------------------------------------------------ params
    def _named_lora(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _trainable(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self._named_lora()]

    def _lora_copy(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self._named_lora()}

    @torch.no_grad()
    def _lora_assign(self, tensors: dict[str, torch.Tensor]) -> None:
        named = dict(self._named_lora())
        if set(named) != set(tensors):
            raise KeyError("LoRA parameter names do not match")
        for n, p in named.items():
            p.copy_(tensors[n].to(device=p.device, dtype=p.dtype))

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
        hidden = self._decoder(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        real = attn.bool()
        h, t = hidden[real], tgt[real]  # [N, H], [N]: only real (unpadded) positions
        grad = torch.is_grad_enabled()
        parts = []
        for s in range(0, h.shape[0], self.logprob_chunk_tokens):
            hc, tc = h[s : s + self.logprob_chunk_tokens], t[s : s + self.logprob_chunk_tokens]
            if grad:
                # Only hc is saved; the chunk's [n, V] logits are recomputed in backward.
                parts.append(checkpoint(self._chunk_logp, hc, tc, use_reentrant=False))
            else:
                parts.append(self._chunk_logp(hc, tc))
        logp = torch.cat(parts)
        return logp.new_zeros(B, T).masked_scatter(real, logp)

    def _chunk_logp(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self._lm_head(h).float()
        return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

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
        self, pairs: Sequence[tuple[list[int], list[int]]], which: Which
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(pairs)
        with torch.no_grad(), self._adapter_mode(which):
            for group in _pack([len(x) for x, _ in pairs], self.micro_batch_tokens):
                logp = self._forward_logp([pairs[i] for i in group])
                for b, i in enumerate(group):
                    out[i] = logp[b, : len(pairs[i][0])].float().cpu()
        return out  # type: ignore[return-value]

    @staticmethod
    def _pair(datum: tinker.Datum) -> tuple[list[int], list[int]]:
        return datum.model_input.to_ints(), _td(datum, "target_tokens", torch.long).tolist()

    def logprobs(self, data: list[tinker.Datum], which: Which) -> list[torch.Tensor]:
        """Per-datum fp32 CPU logprobs of ``target_tokens`` under the chosen weights."""
        return self._logprobs_for([self._pair(d) for d in data], which)

    def as_ref_client(self, which: Which) -> RefLogprobClient:
        return RefLogprobClient(self, which)

    # ------------------------------------------------------------------ training
    def _forward_backward(self, data: list[tinker.Datum], loss_fn: str) -> dict[str, float]:
        """Accumulate token-sum loss grads into ``.grad`` (no zeroing, no step)."""
        loss_sum = 0.0
        n_resp = 0.0
        gap_sum = 0.0
        pairs = [self._pair(d) for d in data]
        self.model.train()  # activates HF gradient checkpointing; dropout is 0
        try:
            for group in _pack([len(x) for x, _ in pairs], self.micro_batch_tokens):
                stats = self._fwd_bwd_group(data, pairs, group, loss_fn)
                loss_sum += stats[0]
                n_resp += stats[1]
                gap_sum += stats[2]
        finally:
            self.model.eval()
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
        if not params:
            norm = torch.zeros(())
        elif clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
        else:
            norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in params])
            )
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.step += 1
        return float(norm)

    def train_step(
        self, data: list[tinker.Datum], loss_fn: str, adam: tinker.AdamParams
    ) -> dict[str, float]:
        if loss_fn not in ("ppo", "importance_sampling"):
            raise ValueError(f"Unsupported loss_fn: {loss_fn!r}")
        self._ensure_optimizer().zero_grad(set_to_none=True)
        metrics = self._forward_backward(data, loss_fn)
        metrics["grad_norm"] = self._optim_step(adam)
        return metrics

    # ------------------------------------------------------------------ adapters / state
    def save_adapter(self, path: str) -> None:
        # save_embedding_layers=False: with lm_head in target_modules PEFT's "auto" would also
        # write the full lm_head.base_layer weight, which vLLM rejects and which is huge.
        self.model.save_pretrained(path, save_embedding_layers=False)

    def snapshot_init_adapter(self) -> None:
        self._init_snapshot = {n: t.cpu() for n, t in self._lora_copy().items()}

    def load_adapter(self, path: str) -> None:
        """Copy adapter weights from a PEFT dir into the live LoRA params; fresh optimizer."""
        cfg = self.model.peft_config["default"]
        saved = PeftConfig.from_pretrained(path)
        if int(saved.r) != int(cfg.r):
            raise ValueError(f"adapter rank {saved.r} != trainer rank {cfg.r}")
        before = self._lora_copy()
        res = set_peft_model_state_dict(self.model, load_peft_weights(path, device="cpu"))
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        unexpected = list(getattr(res, "unexpected_keys", []))
        if missing or unexpected:
            self._lora_assign(before)
            raise RuntimeError(f"adapter mismatch: missing={missing} unexpected={unexpected}")
        self.optimizer = None

    def state_dict_for_ckpt(self) -> dict:
        return {
            "lora": {n: t.cpu() for n, t in self._lora_copy().items()},
            # deepcopy: Optimizer.state_dict() returns references to the live moment tensors.
            "optimizer": None if self.optimizer is None else copy.deepcopy(
                self.optimizer.state_dict()
            ),
            "init_snapshot": None if self._init_snapshot is None else {
                n: t.clone() for n, t in self._init_snapshot.items()
            },
            "step": self.step,
        }

    def load_ckpt_state(self, d: dict) -> None:
        self._lora_assign(d["lora"])
        self.optimizer = None
        if d.get("optimizer") is not None:
            self._ensure_optimizer().load_state_dict(copy.deepcopy(d["optimizer"]))
        snap = d.get("init_snapshot")
        self._init_snapshot = None if snap is None else {n: t.clone() for n, t in snap.items()}
        self.step = int(d.get("step", 0))
