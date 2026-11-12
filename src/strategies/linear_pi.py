# src/strategies/linear_pi.py
from .base_strategy import BaseStrategy
from src.rope import apply_rope_scaling


class RopeScalingStrategy(BaseStrategy):
    """Apply a RoPE scaling method to an already loaded model, without fine-tuning."""

    def __init__(self, method: str, original_length: int, target_length: int):
        super().__init__(method)
        self.method = method
        self.original_length = original_length
        self.target_length = target_length
        self.scaling_factor = target_length / original_length
        self.config = {
            "method": method,
            "original_length": original_length,
            "target_length": target_length,
            "scaling_factor": self.scaling_factor,
        }

    def apply(self, model):
        """`model` is the transformers model. Raises if scaling does not take effect."""
        info = apply_rope_scaling(model, self.method, self.original_length, self.target_length)
        self.config["rope"] = info
        print(f"✓ RoPE scaling applied and verified: {info}")
        return model


class LinearPIStrategy(RopeScalingStrategy):
    """Linear Position Interpolation, zero-shot."""

    def __init__(self, original_length: int, target_length: int):
        super().__init__("linear", original_length, target_length)


class YaRNStrategy(RopeScalingStrategy):
    """YaRN frequencies and attention temperature, zero-shot."""

    def __init__(self, original_length: int, target_length: int):
        super().__init__("yarn", original_length, target_length)
