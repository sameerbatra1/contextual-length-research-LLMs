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
        model_type = model_config["type"]
        model_path = model_config["path"]
        
        # ← ADD THIS BLOCK (support multiple models)
        if model_type == "GemmaModel":
            print("Loading Gemma model...")
            from src.models.gemma_model import GemmaModel
            self.model = GemmaModel()
            self.model.load(model_path)
            self.model.set_seed(self.config["experiment"]["seed"])
        
        elif model_type == "LlamaModel":  # ← NEW
            print("Loading LLaMA model...")
            from src.models.Llama_model import LlamaModel
            self.model = LlamaModel()
            self.model.load(model_path)
            self.model.set_seed(self.config["experiment"]["seed"])
        
        elif model_type == "Phi2Model":  # ← Keep existing
            print("Loading Phi2 model...")
            from src.models.phi2_model import Phi2Model
            self.model = Phi2Model()
            self.model.load(model_path)
            self.model.set_seed(self.config["experiment"]["seed"])
        
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: GemmaModel, LlamaModel, Phi2Model")
        
        # Verify model loaded
        if self.model is None:
            raise RuntimeError(f"Failed to load {model_type}")
        
        print(f"✓ Loaded {model_type}")
        
        # Initialize strategy
        strategy_config = self.config["strategy"]
        if strategy_config["type"] == "LinearPIStrategy":
            from src.strategies.linear_pi import LinearPIStrategy
            self.strategy = LinearPIStrategy(
                original_length=strategy_config["original_length"],
                target_length=strategy_config["target_length"]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_config['type']}")
        
        # Initialize evaluators
        eval_config = self.config["evaluation"]
        if "needle_haystack" in eval_config:
            from src.evaluators.needle_haystack import NeedleHaystackEvaluator
            evaluator = NeedleHaystackEvaluator(eval_config["needle_haystack"])
            self.evaluators.append(evaluator)
        
        print("✓ Setup complete\n")
    
    def run(self):
        """Run the experiment"""
        self.setup()
        
        # Apply strategy
        print("Applying strategy...")
        self.strategy.apply(self.model)
        
        # Run evaluations
        print("Running evaluations...\n")
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
        
        print(f"✓ Results saved to: {filepath}")
