#!/usr/bin/env python3
"""
Fine-tune a model for context extension with a chosen RoPE method.

    python scripts/finetune.py --model TinyLlama/TinyLlama_v1.1 --method linear \
        --data_path data/pile_train_stratified/train_documents.json \
        --output_dir checkpoints/phase0/tinyllama_linear_s42 --seed 42
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rope import METHODS  # noqa: E402
from src.training.continue_pretrain import ContextExtensionTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--original_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true", help="Train in bf16 instead of 4-bit")
    args = parser.parse_args()

    trainer = ContextExtensionTrainer(
        model_name=args.model,
        context_length=args.context_length,
        method=args.method,
        original_length=args.original_length,
        use_4bit=not args.no_4bit,
        seed=args.seed,
    )
    trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
