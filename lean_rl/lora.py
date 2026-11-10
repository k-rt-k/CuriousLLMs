"""LoRA placement matching Tinker's defaults (attention, MLP incl. MoE experts, lm_head).

PEFT's ``all-linear`` excludes ``lm_head``; Tinker's ``create_lora_training_client``
defaults (train_mlp/train_attn/train_unembed=True) include it, so we compute the
target list explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch.nn as nn
from peft import LoraConfig

logger = logging.getLogger(__name__)

LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# MoE routers/gates are linear-like but are never LoRA targets.
_ROUTER_NAMES = {"router", "gate"}
_GPT_OSS_EXPERT_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


@dataclass
class LoraPlacement:
    target_modules: list[str] = field(default_factory=list)
    target_parameters: list[str] = field(default_factory=list)
    includes_lm_head: bool = False
    mismatches: list[str] = field(default_factory=list)


def _is_moe(config) -> bool:
    return any(
        getattr(config, k, None) not in (None, 0, 1)
        for k in ("num_experts", "num_local_experts", "n_routed_experts")
    )


def lora_placement(
    model, *, train_unembed: bool = True, gpt_oss_lm_head: bool = False
) -> LoraPlacement:
    """Compute LoRA targets for ``model`` mirroring Tinker's placement."""
    config = model.config
    model_type = getattr(config, "model_type", "")
    is_gpt_oss = model_type == "gpt_oss"

    names: set[str] = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not full_name:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in _ROUTER_NAMES or leaf == "lm_head":
            continue
        names.add(leaf)

    targets = sorted(names)
    includes_lm_head = False
    mismatches: list[str] = []

    if train_unembed:
        if is_gpt_oss and not gpt_oss_lm_head:
            mismatches.append(
                "lm_head is not given a LoRA adapter for gpt-oss (gpt_oss_lm_head=False), "
                "whereas Tinker's train_unembed=True adapts the unembedding layer"
            )
        else:
            targets.append("lm_head")
            includes_lm_head = True

    target_parameters: list[str] = []
    if is_gpt_oss:
        target_parameters = list(_GPT_OSS_EXPERT_PARAMS)

    if is_gpt_oss or _is_moe(config):
        mismatches.append(
            "per-expert rank: PEFT applies the given LoRA rank to each expert separately, "
            "which may differ from how Tinker splits rank across MoE experts"
        )

    return LoraPlacement(
        target_modules=targets,
        target_parameters=target_parameters,
        includes_lm_head=includes_lm_head,
        mismatches=mismatches,
    )


def make_lora_config(placement: LoraPlacement, rank: int) -> LoraConfig:
    kwargs = {}
    if placement.target_parameters:
        kwargs["target_parameters"] = list(placement.target_parameters)
    return LoraConfig(
        r=rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(placement.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        **kwargs,
    )


def log_parity_warnings(placement: LoraPlacement, model_type: str) -> None:
    for m in placement.mismatches:
        logger.warning("WARNING [tinker-parity] (%s) %s", model_type, m)
