# main.py
"""
Main entry point for context length extension experiments.

Usage:
    python main.py --config configs/experiments/linear_pi_basic.yaml
"""
import argparse
from pathlib import Path
from src.experiment import ExperimentRunner
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(
        description="Context Length Extension Experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Run experiment
    runner = ExperimentRunner(config, output_dir=args.output)
    results = runner.run()
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
