import pytest
import torch
from peft import get_peft_model
from transformers import (
    GptOssConfig, GptOssForCausalLM, LlamaConfig, LlamaForCausalLM,
    Qwen3Config, Qwen3ForCausalLM,
)

from lean_rl.lora import log_parity_warnings, lora_placement, make_lora_config

DENSE = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
_common = dict(hidden_size=64, intermediate_size=64, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, vocab_size=128)


def _llama():
    return LlamaForCausalLM(LlamaConfig(**_common))


def _qwen3():
    return Qwen3ForCausalLM(Qwen3Config(head_dim=16, **_common))


def _gpt_oss():
    return GptOssForCausalLM(GptOssConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=128, num_local_experts=4,
        num_experts_per_tok=2, layer_types=["full_attention", "sliding_attention"]))


@pytest.mark.parametrize("build", [_llama, _qwen3])
def test_dense_targets_all_linear_plus_lm_head(build):
    p = lora_placement(build())
    assert DENSE <= set(p.target_modules)
    assert p.target_parameters == []
    assert p.includes_lm_head
    assert p.mismatches == []


def test_gpt_oss_experts_via_target_parameters():
    p = lora_placement(_gpt_oss())
    assert p.target_parameters == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(p.target_modules)
    assert not {"router", "gate"} & set(p.target_modules)
    assert p.includes_lm_head is False
    assert any("lm_head" in m for m in p.mismatches)
    assert any("per-expert rank" in m for m in p.mismatches)
    assert lora_placement(_gpt_oss(), gpt_oss_lm_head=True).includes_lm_head


def test_peft_model_trains_expected_params():
    cfg_p = lora_placement(_llama())
    cfg = make_lora_config(cfg_p, 8)
    assert cfg.lora_alpha == 32 and cfg.lora_dropout == 0.0
    peft = get_peft_model(_llama(), cfg)
    names = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert any("lm_head" in n for n in names)

    model = _gpt_oss()
    peft = get_peft_model(model, make_lora_config(lora_placement(model), 8))
    names = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert any("experts" in n for n in names)
    assert not any("router" in n for n in names)
    out = peft(input_ids=torch.randint(0, 128, (1, 8)))
    assert out.logits.shape[-1] == 128


def test_train_unembed_false_excludes_lm_head():
    p = lora_placement(_llama(), train_unembed=False)
    assert "lm_head" not in p.target_modules and not p.includes_lm_head
    assert p.mismatches == []


def test_log_parity_warnings(caplog):
    p = lora_placement(_gpt_oss())
    with caplog.at_level("WARNING"):
        log_parity_warnings(p, "gpt_oss")
    assert sum("[tinker-parity]" in r.message for r in caplog.records) == len(p.mismatches)
