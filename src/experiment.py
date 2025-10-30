# src/experiment.py
from pathlib import Path
import json
from datetime import datetime

class ExperimentRunner:
    """Orchestrates experiments"""
    
    def __init__(self, config: dict, output_dir: str = "results/"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.strategy = None
        self.evaluators = []
    
    def setup(self):
        """Initialize model, strategy, and evaluators"""
        print("Setting up experiment...")
        
        # Initialize model
        model_config = self.config["model"]
        if model_config["type"] == "GemmaModel":
            from src.models.gemma_model import GemmaModel
            self.model = GemmaModel()
            self.model.load(model_config["path"])
            self.model.set_seed(self.config["experiment"]["seed"])
        
        # Initialize strategy
        strategy_config = self.config["strategy"]
        if strategy_config["type"] == "LinearPIStrategy":
            from src.strategies.linear_pi import LinearPIStrategy
            self.strategy = LinearPIStrategy(
                original_length=strategy_config["original_length"],
                target_length=strategy_config["target_length"]
            )
        
        # Initialize evaluators
        eval_config = self.config["evaluation"]
        if "needle_haystack" in eval_config:
            from src.evaluators.needle_haystack import NeedleHaystackEvaluator
            evaluator = NeedleHaystackEvaluator(eval_config["needle_haystack"])
            self.evaluators.append(evaluator)
        
        print("✓ Setup complete")
    
    def run(self):
        """Run the experiment"""
        self.setup()
        
        # Apply strategy
        self.strategy.apply(self.model)
        
        # Run evaluations
        all_results = {}
        for evaluator in self.evaluators:
            results = evaluator.evaluate(self.model)
            all_results[evaluator.name] = results
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: dict):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['experiment']['name']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        output = {
            "config": self.config,
            "results": results,
            "timestamp": timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
