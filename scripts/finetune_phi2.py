#!/usr/bin/env python3

import argparse
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root (parent of scripts directory)
project_root = os.path.dirname(script_dir)

# Add project root to Python path
sys.path.insert(0, project_root)

from src.training.continue_pretrain import ContextExtensionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to PILE JSONL file")
    parser.add_argument("--output_dir", default="checkpoints/phi2_yarn_g5", help="Output directory")
    parser.add_argument("--context_length", type=int, default=16384, help="Target context length")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ContextExtensionTrainer(
        model_name="microsoft/phi-2",
        context_length=args.context_length
    )
    
    # Train
    trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
