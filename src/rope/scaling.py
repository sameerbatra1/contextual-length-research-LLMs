"""
Single entry point for RoPE scaling.

Every code path that extends a model's context (training, evaluation of base
models, evaluation of fine-tuned adapters) goes through `apply_rope_scaling`,
so the position encoding used at evaluation is the one used in training.

The function rebuilds the model's rotary-embedding module from its config and
then checks the resulting frequencies against the unscaled ones. It raises if
the scaling did not take effect, instead of silently leaving the model
unchanged.

Methods
-------
none     Unmodified RoPE (plain extrapolation beyond the native window).
linear   Position Interpolation (Chen et al., 2023): every frequency / s.
yarn     YaRN (Peng et al., 2024): NTK-by-parts + attention temperature.
dynamic  Dynamic NTK as implemented in transformers. Depends on the length of
         each input. Kept only to reproduce the pilot runs.
"""

import copy
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

METHODS = ("none", "linear", "yarn", "dynamic")
EXTENSION_CONFIG_NAME = "extension_config.json"


def _rotary_modules(model: nn.Module) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Return (parent, attribute name, module) for every rotary embedding."""
    found = []
    for name, module in model.named_modules():
        if name.split(".")[-1] != "rotary_emb":
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        found.append((parent, attr, module))
    return found


def build_rope_scaling(method: str, original_length: int, target_length: int):
    """Return (rope_scaling dict or None, max_position_embeddings) for a method."""
    if method not in METHODS:
        raise ValueError(f"Unknown RoPE method '{method}'. Choose one of {METHODS}.")
    factor = target_length / original_length
    if method == "none" or factor == 1.0:
        return None, original_length
    if method == "linear":
        return {"rope_type": "linear", "factor": factor}, target_length
    if method == "yarn":
        return (
            {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_length,
            },
            target_length,
        )
    # dynamic: transformers rescales from max_position_embeddings, so keep it native
    return {"rope_type": "dynamic", "factor": factor}, original_length


def _default_inv_freq(config, device) -> torch.Tensor:
    reference = copy.deepcopy(config)
    reference.rope_scaling = None
    inv_freq, _ = ROPE_INIT_FUNCTIONS["default"](reference, device)
    return inv_freq


def apply_rope_scaling(
    model: nn.Module,
    method: str,
    original_length: int,
    target_length: int,
) -> Dict[str, Any]:
    """
    Apply a RoPE scaling method to a loaded transformers model, in place.

    Returns a description of what was applied (stored with results so every
    result file records the position encoding that produced it).
    """
    rope_scaling, max_positions = build_rope_scaling(method, original_length, target_length)
    factor = target_length / original_length

    modules = _rotary_modules(model)
    if not modules:
        raise RuntimeError(
            "No 'rotary_emb' module found on the model, so RoPE scaling cannot be "
            "applied. Check the transformers version and the model architecture."
        )

    config = model.config
    config.rope_scaling = rope_scaling
    config.max_position_embeddings = max_positions

    for parent, attr, old in modules:
        device = old.inv_freq.device
        setattr(parent, attr, type(old)(config=config, device=device))

    return verify_rope_scaling(model, method, factor)


def verify_rope_scaling(model: nn.Module, method: str, factor: float) -> Dict[str, Any]:
    """Check the model's rotary frequencies match `method`; raise if they do not."""
    modules = _rotary_modules(model)
    if not modules:
        raise RuntimeError("No 'rotary_emb' module found on the model.")

    description = None
    for _, _, module in modules:
        inv_freq = module.inv_freq.float().cpu()
        base = _default_inv_freq(model.config, "cpu").float()
        if inv_freq.shape != base.shape:
            raise RuntimeError(
                f"Rotary dimension mismatch: model has {tuple(inv_freq.shape)}, "
                f"expected {tuple(base.shape)}."
            )
        ratio = inv_freq / base
        attention_scaling = float(getattr(module, "attention_scaling", 1.0))
        rope_type = getattr(module, "rope_type", "default")

        def close(a, b):
            return torch.allclose(a, b, rtol=1e-4, atol=0)

        ones = torch.ones_like(ratio)
        if method == "none" or factor == 1.0:
            ok = close(ratio, ones) and attention_scaling == 1.0
        elif method == "linear":
            ok = close(ratio, ones / factor) and attention_scaling == 1.0
        elif method == "yarn":
            expected_attention = 0.1 * math.log(factor) + 1.0
            ok = (
                close(ratio[:1], ones[:1])                 # highest frequency untouched
                and close(ratio[-1:], ones[-1:] / factor)  # lowest frequency fully interpolated
                and bool((ratio[1:] <= ratio[:-1] + 1e-6).all())
                and abs(attention_scaling - expected_attention) < 1e-4
            )
        else:  # dynamic: frequencies change per input length, only the type can be checked
            ok = rope_type == "dynamic"

        if not ok:
            raise RuntimeError(
                f"RoPE scaling check failed for method '{method}' (factor {factor}): "
                f"rope_type={rope_type}, inv_freq ratio first={ratio[0]:.4f} "
                f"last={ratio[-1]:.4f}, attention_scaling={attention_scaling:.4f}."
            )

        description = {
            "method": method,
            "factor": factor,
            "rope_type": rope_type,
            "rotary_dim": int(inv_freq.numel() * 2),
            "inv_freq_ratio_first": float(ratio[0]),
            "inv_freq_ratio_last": float(ratio[-1]),
            "attention_scaling": attention_scaling,
            "num_rotary_modules": len(modules),
            "max_position_embeddings": int(model.config.max_position_embeddings),
        }
    return description
