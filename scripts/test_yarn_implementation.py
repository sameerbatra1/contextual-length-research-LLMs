#!/usr/bin/env python3
"""
Test YaRN implementation before training

Quick validation script to verify YaRN is correctly implemented
following the paper specifications.
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.rope import (
    compute_yarn_frequencies,
    compute_mscale,
    get_dynamic_scale,
    YaRNRotaryEmbedding,
)


def test_yarn_frequencies():
    """Test YaRN frequency computation"""
    print("\n" + "="*70)
    print("Test 1: YaRN Frequency Computation")
    print("="*70)
    
    dim = 64
    max_pos = 2048
    scale = 4.0
    alpha = 1.0
    beta = 32.0
    
    inv_freq = compute_yarn_frequencies(
        dim=dim,
        max_position_embeddings=max_pos,
        base=10000.0,
        scale=scale,
        alpha=alpha,
        beta=beta,
    )
    
    print(f"Parameters: dim={dim}, max_pos={max_pos}, scale={scale}x")
    print(f"Alpha (α): {alpha}, Beta (β): {beta}")
    print(f"Result shape: {inv_freq.shape}")
    print(f"Result range: [{inv_freq.min():.6f}, {inv_freq.max():.6f}]")
    
    # Verify NTK-by-parts: low frequencies should be scaled more
    standard_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    ratio_low = inv_freq[0] / standard_inv_freq[0]
    ratio_high = inv_freq[-1] / standard_inv_freq[-1]
    
    print(f"\nScaling verification:")
    print(f"  Low freq ratio:  {ratio_low:.4f} (should be ~{1/scale:.4f})")
    print(f"  High freq ratio: {ratio_high:.4f} (should be ~1.0)")
    
    if ratio_low < ratio_high:
        print("✅ PASS: Low frequencies scaled more than high frequencies")
        return True
    else:
        print("❌ FAIL: Incorrect frequency scaling")
        return False


def test_mscale():
    """Test attention temperature scaling"""
    print("\n" + "="*70)
    print("Test 2: Attention Temperature Scaling (mscale)")
    print("="*70)
    
    test_cases = [
        (1.0, 1.0),
        (2.0, 0.1 * np.log(2) + 1),
        (4.0, 0.1 * np.log(4) + 1),
        (16.0, 0.1 * np.log(16) + 1),
    ]
    
    print(f"\n{'Scale':>8s} {'Expected':>12s} {'Computed':>12s} {'Status':>8s}")
    print("-" * 45)
    
    all_pass = True
    for scale, expected in test_cases:
        computed = compute_mscale(scale)
        status = "✅" if abs(computed - expected) < 1e-6 else "❌"
        if status == "❌":
            all_pass = False
        print(f"{scale:>8.1f} {expected:>12.4f} {computed:>12.4f} {status:>8s}")
    
    if all_pass:
        print("\n✅ PASS: mscale computation correct")
    else:
        print("\n❌ FAIL: mscale computation incorrect")
    
    return all_pass


def test_dynamic_scaling():
    """Test dynamic scaling"""
    print("\n" + "="*70)
    print("Test 3: Dynamic Scaling")
    print("="*70)
    
    max_pos = 2048
    test_cases = [
        (1024, 1.0),
        (2048, 1.0),
        (4096, 2.0),
        (8192, 4.0),
    ]
    
    print(f"\n{'Seq Len':>10s} {'Expected':>12s} {'Computed':>12s} {'Status':>8s}")
    print("-" * 45)
    
    all_pass = True
    for seq_len, expected in test_cases:
        computed = get_dynamic_scale(seq_len, max_pos)
        status = "✅" if abs(computed - expected) < 1e-6 else "❌"
        if status == "❌":
            all_pass = False
        print(f"{seq_len:>10d} {expected:>12.2f} {computed:>12.2f} {status:>8s}")
    
    if all_pass:
        print("\n✅ PASS: Dynamic scaling correct")
    else:
        print("\n❌ FAIL: Dynamic scaling incorrect")
    
    return all_pass


def test_yarn_embedding():
    """Test YaRNRotaryEmbedding module"""
    print("\n" + "="*70)
    print("Test 4: YaRNRotaryEmbedding Module")
    print("="*70)
    
    head_dim = 64
    max_pos = 2048
    scaling_factor = 4.0
    batch_size = 2
    seq_len = 1024
    
    yarn_emb = YaRNRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=max_pos,
        scaling_factor=scaling_factor,
        alpha=1.0,
        beta=32.0,
    )
    
    print(f"Parameters: head_dim={head_dim}, max_pos={max_pos}, scale={scaling_factor}x")
    print(f"Module: {yarn_emb}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, 1, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    try:
        with torch.no_grad():
            cos, sin = yarn_emb(x, position_ids)
        
        print(f"\nForward pass results:")
        print(f"  Input shape:  {x.shape}")
        print(f"  cos shape:    {cos.shape}")
        print(f"  sin shape:    {sin.shape}")
        print(f"  cos range:    [{cos.min():.4f}, {cos.max():.4f}]")
        print(f"  sin range:    [{sin.min():.4f}, {sin.max():.4f}]")
        
        # Verify shapes
        if cos.shape == (batch_size, seq_len, head_dim) and sin.shape == (batch_size, seq_len, head_dim):
            print("\n✅ PASS: YaRNRotaryEmbedding working correctly")
            return True
        else:
            print("\n❌ FAIL: Incorrect output shapes")
            return False
            
    except Exception as e:
        print(f"\n❌ FAIL: Error during forward pass: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("YaRN Implementation Test Suite")
    print("Based on ICLR 2024 paper by Peng et al.")
    print("="*70)
    
    tests = [
        ("YaRN Frequency Computation", test_yarn_frequencies),
        ("Attention Temperature Scaling", test_mscale),
        ("Dynamic Scaling", test_dynamic_scaling),
        ("YaRNRotaryEmbedding Module", test_yarn_embedding),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! YaRN implementation is correct.")
        print("\nYou can now proceed with training:")
        print("  python scripts/finetune_phi2_yarn.py \\")
        print("      --data_path data/pile_train_stratified/train_documents.json \\")
        print("      --output_dir checkpoints/phi2_yarn_8k \\")
        print("      --context_length 8192")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

