#!/usr/bin/env python3
"""
Fine-tune Phi-2 with YaRN Context Extension

Implementation following ICLR 2024 paper specifications:
"YaRN: Efficient Context Window Extension of Large Language Models"

Usage:
    python scripts/finetune_phi2_yarn.py \
        --data_path data/pile_train_stratified/train_documents.json \
        --output_dir checkpoints/phi2_yarn_8k \
        --context_length 8192

    python scripts/finetune_phi2_yarn.py \
        --data_path data/pile_train_stratified/train_documents.json \
        --output_dir checkpoints/phi2_yarn_32k \
        --context_length 32768

Paper: https://arxiv.org/abs/2309.00071
"""

import argparse
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root (parent of scripts directory)
project_root = os.path.dirname(script_dir)

# Add project root to Python path
sys.path.insert(0, project_root)

from src.training.yarn_trainer import YaRNTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-2 with YaRN context extension"
    )
    
    # Required arguments
    parser.add_argument(
        "--data_path", 
        required=True, 
        help="Path to training data (JSONL file)"
    )
    parser.add_argument(
        "--output_dir", 
        default="checkpoints/phi2_yarn", 
        help="Output directory for model checkpoints"
    )
    
    # Context extension parameters
    parser.add_argument(
        "--context_length", 
        type=int, 
        default=8192,
        choices=[4096, 8192, 16384, 32768, 65536],
        help="Target context length (paper tested: 8k, 16k, 32k, 64k)"
    )
    
    # YaRN hyperparameters (paper defaults)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="YaRN alpha parameter (NTK lower bound, paper: 1.0)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=32.0,
        help="YaRN beta parameter (NTK upper bound, paper: 32.0)"
    )
    parser.add_argument(
        "--dynamic_scaling",
        action="store_true",
        help="Enable dynamic scaling at inference time"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more memory)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("YaRN Fine-tuning for Phi-2")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Context: 2048 → {args.context_length} ({args.context_length/2048:.1f}x)")
    print(f"Alpha (α): {args.alpha}")
    print(f"Beta (β): {args.beta}")
    print(f"4-bit: {not args.no_4bit}")
    print(f"Dynamic scaling: {args.dynamic_scaling}")
    print("="*80 + "\n")
    
    # Validate context length
    if args.context_length < 2048:
        print("❌ Error: Context length must be at least 2048")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize YaRN trainer
    print("Initializing YaRN trainer...")
    trainer = YaRNTrainer(
        model_name=args.model_name,
        context_length=args.context_length,
        alpha=args.alpha,
        beta=args.beta,
        use_dynamic_scaling=args.dynamic_scaling,
        use_4bit=not args.no_4bit,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Print completion message
    print("\n" + "="*80)
    print("✅ YaRN Fine-tuning Complete!")
    print("="*80)
    print(f"\n📁 Model saved to: {args.output_dir}/final_model")
    print(f"📁 YaRN config saved to: {args.output_dir}/yarn_config.json")
    print(f"\n🔬 Next steps:")
    print(f"   1. Evaluate with needle-in-haystack benchmark")
    print(f"   2. Test perplexity on long documents")
    print(f"   3. Compare with Linear PI baseline")
    print(f"\n📚 Paper reference:")
    print(f"   Peng et al., 'YaRN: Efficient Context Window Extension of LLMs'")
    print(f"   ICLR 2024, https://arxiv.org/abs/2309.00071")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

