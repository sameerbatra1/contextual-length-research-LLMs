"""
CPU-only checks that RoPE scaling really changes the model.

Uses tiny randomly initialised Llama- and Phi-shaped models, so no weights are
downloaded. Run with:  python -m pytest tests/ -q
"""
import math
import os
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, PhiConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rope import apply_rope_scaling, build_rope_scaling  # noqa: E402
from src.rope.yarn_rope import replace_phi_rope_with_yarn  # noqa: E402

ORIGINAL, TARGET = 2048, 8192
FACTOR = TARGET / ORIGINAL


def tiny(kind, **config_kwargs):
    common = dict(vocab_size=100, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4,
                  max_position_embeddings=ORIGINAL, **config_kwargs)
    if kind == "llama":
        config = LlamaConfig(num_key_value_heads=4, **common)
    else:
        config = PhiConfig(partial_rotary_factor=0.5, **common)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).eval()


def logits(model, n_tokens=300):
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (1, n_tokens))
    with torch.no_grad():
        return model(ids).logits


KINDS = ["llama", "phi"]


@pytest.mark.parametrize("kind", KINDS)
def test_none_leaves_model_unchanged(kind):
    reference = logits(tiny(kind))
    model = tiny(kind)
    info = apply_rope_scaling(model, "none", ORIGINAL, TARGET)
    assert info["inv_freq_ratio_first"] == pytest.approx(1.0)
    assert torch.allclose(logits(model), reference)


@pytest.mark.parametrize("kind", KINDS)
def test_linear_divides_every_frequency(kind):
    reference = logits(tiny(kind))
    model = tiny(kind)
    info = apply_rope_scaling(model, "linear", ORIGINAL, TARGET)
    assert info["inv_freq_ratio_first"] == pytest.approx(1 / FACTOR, rel=1e-4)
    assert info["inv_freq_ratio_last"] == pytest.approx(1 / FACTOR, rel=1e-4)
    assert info["attention_scaling"] == 1.0
    assert model.config.max_position_embeddings == TARGET
    assert not torch.allclose(logits(model), reference)


@pytest.mark.parametrize("kind", KINDS)
def test_yarn_scales_by_parts(kind):
    reference = logits(tiny(kind))
    model = tiny(kind)
    info = apply_rope_scaling(model, "yarn", ORIGINAL, TARGET)
    assert info["inv_freq_ratio_first"] == pytest.approx(1.0, rel=1e-4)
    assert info["inv_freq_ratio_last"] == pytest.approx(1 / FACTOR, rel=1e-4)
    assert info["attention_scaling"] == pytest.approx(0.1 * math.log(FACTOR) + 1)
    assert not torch.allclose(logits(model), reference)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["linear", "yarn"])
def test_post_load_matches_load_time_config(kind, method):
    """Rebuilding after load must equal passing rope_scaling when the model is built."""
    rope_scaling, max_positions = build_rope_scaling(method, ORIGINAL, TARGET)
    built = tiny(kind, rope_scaling=rope_scaling)
    built.config.max_position_embeddings = max_positions
    patched = tiny(kind)
    apply_rope_scaling(patched, method, ORIGINAL, TARGET)
    assert torch.allclose(logits(patched), logits(built), atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_phi_partial_rotary_dimension_is_respected(kind):
    model = tiny(kind)
    info = apply_rope_scaling(model, "yarn", ORIGINAL, TARGET)
    head_dim = 64 // 4
    expected = head_dim // 2 if kind == "phi" else head_dim
    assert info["rotary_dim"] == expected


@pytest.mark.parametrize("kind", KINDS)
def test_pilot_bug_setting_config_after_load_is_a_noop(kind):
    """Documents the pilot's base-run bug: config changes alone do nothing."""
    reference = logits(tiny(kind))
    model = tiny(kind)
    model.config.rope_scaling = {"type": "linear", "factor": FACTOR}
    model.config.max_position_embeddings = TARGET
    assert torch.allclose(logits(model), reference)


@pytest.mark.parametrize("kind", KINDS)
def test_pilot_yarn_replacement_now_fails_loudly(kind):
    """The pilot's per-layer YaRN swap finds no layers on this transformers version."""
    with pytest.raises(RuntimeError, match="replaced 0 layers"):
        replace_phi_rope_with_yarn(tiny(kind), scaling_factor=FACTOR)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        apply_rope_scaling(tiny("llama"), "magic", ORIGINAL, TARGET)
