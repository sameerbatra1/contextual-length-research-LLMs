# src/strategies/linear_pi.py
from .base_strategy import BaseStrategy

class LinearPIStrategy(BaseStrategy):
    """Linear Position Interpolation strategy"""
    
    def __init__(self, original_length: int, target_length: int):
        super().__init__("linear_pi")
        
        self.original_length = original_length
        self.target_length = target_length
        self.scaling_factor = target_length / original_length
        
        self.config = {
            "original_length": original_length,
            "target_length": target_length,
            "scaling_factor": self.scaling_factor
        }
    
    def apply(self, model):
        """Apply Linear PI to model"""
        print(f"\nApplying Linear PI: {self.original_length} → {self.target_length}")
        print(f"Scaling factor: {self.scaling_factor:.2f}")
        
        rope_config = {
            "type": "linear",
            "factor": self.scaling_factor
        }
        
        # Apply through model's interface
        if hasattr(model, 'apply_rope_scaling'):
            model.apply_rope_scaling(rope_config)
        else:
            # Direct access if method not available
            model.model.config.rope_scaling = rope_config
        
        print("✓ Linear PI applied successfully")
        return model
