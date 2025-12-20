"""
YaRN: Yet another RoPE extensioN method
Official implementation based on ICLR 2024 paper by Peng et al.

Paper: "YaRN: Efficient Context Window Extension of Large Language Models"
Reference: https://github.com/jquesnelle/yarn
arXiv: https://arxiv.org/abs/2309.00071
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple


def compute_yarn_frequencies(
    dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    scale: float = 1.0,
    alpha: float = 1.0,
    beta: float = 32.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute YaRN modified RoPE frequencies using NTK-by-parts interpolation.
    
    The key formula modifies RoPE frequencies dimension-wise:
    θ'_d = [(1 - γ(r(d))) · θ_d/s] + γ(r(d)) · θ_d
    
    Where:
    - r(d) = L / (2π · b^(2d/D)) is the ratio of context length to wavelength
    - γ(r) is the ramp function that determines interpolation amount per dimension
    
    Args:
        dim: Hidden dimension size (D)
        max_position_embeddings: Original context length (L)
        base: RoPE base (b), typically 10000.0
        scale: Extension scale factor (s = L'/L)
        alpha: Lower bound for ramp function (α)
        beta: Upper bound for ramp function (β)
        device: Device to place tensors on
        
    Returns:
        inv_freq_scaled: YaRN-modified inverse frequencies
    """
    # Standard RoPE inverse frequencies: 1 / (base^(2d/D))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    
    # Compute wavelength for each dimension: λ = 2π / inv_freq
    wavelength = 2 * np.pi / inv_freq
    
    # Compute wavelength ratio r(d) = L / λ_d for each dimension
    r = max_position_embeddings / wavelength
    
    # Compute ramp function γ(r)
    gamma = torch.zeros_like(r)
    
    # Three regions:
    # 1. r < α (low frequencies): γ = 0 → Full interpolation (scale by 1/s)
    # 2. α < r < β (mid frequencies): γ = (r - α)/(β - α) → Linear ramp
    # 3. r > β (high frequencies): γ = 1 → No interpolation (keep original)
    mask_low = r < alpha
    mask_high = r > beta
    mask_mid = ~(mask_low | mask_high)
    
    gamma[mask_low] = 0.0
    gamma[mask_high] = 1.0
    gamma[mask_mid] = (r[mask_mid] - alpha) / (beta - alpha)
    
    # Apply NTK-by-parts scaling: θ'_d = (1-γ)·θ_d/s + γ·θ_d
    inv_freq_scaled = (1 - gamma) * (inv_freq / scale) + gamma * inv_freq
    
    return inv_freq_scaled


def compute_mscale(scale: float, mscale_constant: float = 0.1) -> float:
    """
    Compute attention temperature factor sqrt(1/t).
    
    This implements the "length scaling trick" which scales RoPE embeddings
    by sqrt(1/t) instead of directly modifying attention computation.
    
    For LLaMA/Llama2/TinyLlama/Phi: sqrt(1/t) = 0.1*ln(s) + 1
    
    Args:
        scale: Extension scale factor (s = L'/L)
        mscale_constant: Constant for mscale formula (default: 0.1)
        
    Returns:
        mscale: Attention temperature scaling factor
    """
    if scale <= 1.0:
        return 1.0
    return mscale_constant * math.log(scale) + 1.0


def get_dynamic_scale(seq_len: int, max_position_embeddings: int) -> float:
    """
    Compute dynamic scaling factor based on actual sequence length.
    
    Dynamic scaling: s = max(1, l'/L)
    where l' is the current sequence length and L is the original context length.
    
    Args:
        seq_len: Current sequence length
        max_position_embeddings: Original context length (L)
        
    Returns:
        scale: Dynamic scale factor
    """
    return max(1.0, seq_len / max_position_embeddings)


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN Rotary Position Embedding with NTK-by-parts interpolation.
    
    This is a drop-in replacement for standard RoPE that enables efficient
    context window extension with minimal fine-tuning.
    
    Paper: "YaRN: Efficient Context Window Extension of Large Language Models"
           Peng et al., ICLR 2024
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        alpha: float = 1.0,
        beta: float = 32.0,
        use_dynamic_scaling: bool = False,
        mscale_constant: float = 0.1,
    ):
        """
        Initialize YaRN Rotary Embedding.
        
        Args:
            dim: Dimension of each attention head (hidden_size / num_heads)
            max_position_embeddings: Original context length (L), typically 2048
            base: RoPE base frequency (b), typically 10000.0
            device: Device to place tensors on
            scaling_factor: Static extension scale factor (s = L'/L)
            alpha: Lower bound for NTK ramp function (default: 1.0)
            beta: Upper bound for NTK ramp function (default: 32.0)
            use_dynamic_scaling: Whether to use dynamic scaling at inference time
            mscale_constant: Constant for attention temperature (default: 0.1)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        self.beta = beta
        self.use_dynamic_scaling = use_dynamic_scaling
        self.mscale_constant = mscale_constant
        
        # Compute YaRN frequencies with static scaling
        inv_freq = compute_yarn_frequencies(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scale=scaling_factor,
            alpha=alpha,
            beta=beta,
            device=device,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Compute mscale (attention temperature) for static scaling
        self.mscale = compute_mscale(scaling_factor, mscale_constant)
        
    @torch.no_grad()
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute rotary embeddings.
        
        Args:
            x: Input tensor (for device/dtype reference)
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            cos, sin: Rotary embeddings to apply to queries and keys
        """
        device = x.device
        dtype = x.dtype
        
        # Ensure position_ids is on correct device
        if position_ids.device != device:
            position_ids = position_ids.to(device)
        
        # Determine sequence length
        seq_len = position_ids.max().item() + 1
        
        # Dynamic scaling: recompute frequencies based on actual sequence length
        if self.use_dynamic_scaling and seq_len > self.max_position_embeddings:
            dynamic_scale = get_dynamic_scale(seq_len, self.max_position_embeddings)
            inv_freq = compute_yarn_frequencies(
                dim=self.dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.base,
                scale=dynamic_scale,
                alpha=self.alpha,
                beta=self.beta,
                device=device,
            )
            mscale = compute_mscale(dynamic_scale, self.mscale_constant)
        else:
            inv_freq = self.inv_freq.to(device)
            mscale = self.mscale
        
        # Compute position encodings
        # inv_freq_expanded: [batch_size, dim/2, 1]
        # position_ids_expanded: [batch_size, 1, seq_len]
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Force float32 for precision
        device_type = device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [batch_size, dim/2, seq_len]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [batch_size, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Apply mscale (temperature scaling trick)
            cos = emb.cos() * mscale
            sin = emb.sin() * mscale
        
        return cos.to(dtype), sin.to(dtype)


def replace_phi_rope_with_yarn(
    model: nn.Module,
    scaling_factor: float = 1.0,
    alpha: float = 1.0,
    beta: float = 32.0,
    use_dynamic_scaling: bool = False,
) -> nn.Module:
    """
    Replace RoPE embeddings with YaRN embeddings for Phi-2, Llama, or TinyLlama models.
    
    Args:
        model: Model (Phi-2, Llama, or TinyLlama)
        scaling_factor: Context extension scale factor (s = L'/L)
        alpha: YaRN alpha parameter (default: 1.0)
        beta: YaRN beta parameter (default: 32.0)
        use_dynamic_scaling: Whether to use dynamic scaling at inference
        
    Returns:
        model: Model with YaRN embeddings
        replaced_count: Number of RoPE layers replaced
    """
    config = model.config
    
    # Get model-specific parameters
    head_dim = config.hidden_size // config.num_attention_heads
    max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    
    replaced_count = 0
    
    # Determine model structure
    # Phi-2: model.layers[i].self_attn.rotary_emb
    # Llama/TinyLlama: model.model.layers[i].self_attn.rotary_emb
    layers = None
    if hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    
    if layers is not None:
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                yarn_emb = YaRNRotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor,
                    alpha=alpha,
                    beta=beta,
                    use_dynamic_scaling=use_dynamic_scaling,
                )
                layer.self_attn.rotary_emb = yarn_emb
                replaced_count += 1
    
    return model, replaced_count

