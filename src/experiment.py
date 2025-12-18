# src/experiment.py
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates experiments with support for multiple models and strategies"""
    
    def __init__(self, config: dict, output_dir: str = "results/"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.strategy = None
        self.evaluators = []
        self.evaluation_config = {}
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to file and console"""
        log_dir = Path(self.config.get('logging', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / self.config.get('logging', {}).get('log_file', 'experiment.txt')
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def setup(self):
        """Initialize model and strategy (evaluators created later with model)"""
        logger.info("="*80)
        logger.info(f"EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("="*80)
        logger.info("Setting up experiment...\n")
        
        self._setup_model()
        self._setup_strategy()
        self._setup_evaluators()
        
        logger.info("✓ Setup complete\n")
    
    def _setup_model(self):
        """Load model based on type in config"""
        model_config = self.config["model"]
        model_type = model_config["type"]
        
        logger.info(f"Loading model: {model_type}")
        
        try:
            if model_type == "Phi2Model":
                from src.models.phi2_model import Phi2Model
                self.model = Phi2Model()
                
                # Get target context length from strategy
                context_length = 2048
                if self.config.get("strategy"):
                    context_length = self.config["strategy"].get("target_length", 2048)
                
                self.model.load(
                    model_path=model_config["path"],
                )
            
            elif model_type == "Phi2FinetunedModel":
                from src.models.phi2_finetuned_model import Phi2FinetunedModel
                self.model = Phi2FinetunedModel()
                self.model.load(
                    model_path=model_config.get("base_path", "microsoft/phi-2"),
                    adapter_path=model_config.get("adapter_path"),
                    context_length=self.config.get("strategy", {}).get("target_length", 16384)
                )
            
            elif model_type == "tinyLlamaModel":
                from src.models.tinyLlama_model import TinyLlamaModel
                self.model = TinyLlamaModel()
                
                # Get target context length from strategy
                context_length = 2048
                if self.config.get("strategy"):
                    context_length = self.config["strategy"].get("target_length", 2048)
                
                self.model.load(
                    model_path=model_config["path"],
                )
            elif model_type == "TinyLlamaFinetunedModel":
                from src.models.tinyLlama_finetuned_model import TinyLlamaFinetunedModel
                self.model = TinyLlamaFinetunedModel()
                self.model.load(
                    model_path=model_config.get("base_path", "microsoft/phi-2"),
                    adapter_path=model_config.get("adapter_path"),
                    context_length=self.config.get("strategy", {}).get("target_length", 16384)
                )

            elif model_type == "GemmaModel":
                from src.models.gemma_model import GemmaModel
                self.model = GemmaModel()
                self.model.load(model_config["path"])
                
            elif model_type == "LlamaModel":
                from src.models.Llama_model import LlamaModel
                self.model = LlamaModel()
                self.model.load(model_config["path"])
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if self.model is None:
                raise RuntimeError(f"Failed to load {model_type}")
            
            logger.info(f"✓ Loaded {model_type}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    def _setup_strategy(self):
        """Initialize context extension strategy"""
        strategy_config = self.config.get("strategy")
        
        if strategy_config is None:
            logger.info("ℹ No strategy configured\n")
            self.strategy = None
            return
        
        strategy_type = strategy_config.get("type")
        
        if strategy_type is None:
            logger.info("ℹ Strategy type is null\n")
            self.strategy = None
            return
        
        logger.info(f"Setting up strategy: {strategy_type}")
        
        try:
            if strategy_type == "LinearPIStrategy":
                from src.strategies.linear_pi import LinearPIStrategy
                self.strategy = LinearPIStrategy(
                    original_length=strategy_config["original_length"],
                    target_length=strategy_config["target_length"]
                )
            
            # elif strategy_type == "YaRNStrategy":
            #     from src.strategies.yarn import YaRNStrategy
            #     self.strategy = YaRNStrategy(
            #         original_length=strategy_config["original_length"],
            #         target_length=strategy_config["target_length"]
            #     )
            
            else:
                raise ValueError(f"Unknown strategy: {strategy_type}")
            
            logger.info(
                f"✓ Strategy: {strategy_type} "
                f"({strategy_config['original_length']} → {strategy_config['target_length']})\n"
            )
        
        except Exception as e:
            logger.error(f"✗ Failed to setup strategy: {e}")
            raise
    
    def _setup_evaluators(self):
        """Store evaluation config (create evaluators later with model)"""
        eval_config = self.config.get("evaluation", {})
        
        logger.info("Evaluation config:")
        for key in eval_config.keys():
            logger.info(f"  - {key}")
        
        self.evaluation_config = eval_config
        logger.info("Evaluation config stored\n")
    
    def _create_evaluators(self):
        """Create evaluators now that model is available"""
        eval_config = self.evaluation_config
        
        try:
            if "needle_haystack" in eval_config:
                from src.evaluators.needle_haystack import NeedleHaystackEvaluator
                
                needle_cfg = eval_config["needle_haystack"]
                evaluator = NeedleHaystackEvaluator(
                    config=needle_cfg
                )
                self.evaluators.append(evaluator)
                logger.info(f"✓ Created NeedleHaystackEvaluator")
            
            if not self.evaluators:
                logger.warning("⚠ No evaluators created")
        
        except Exception as e:
            logger.error(f"✗ Failed to create evaluators: {e}")
            raise
    
    def run(self):
        """Run the complete experiment"""
        logger.info("="*80)
        logger.info("STARTING EXPERIMENT")
        logger.info("="*80 + "\n")
        
        try:
            # Setup
            self.setup()
            
            # Skip strategy if using fine-tuned model
            model_type = self.config["model"]["type"]
            if model_type == "Phi2FinetunedModel":
                logger.info("ℹ Skipping strategy (RoPE scaling already applied during model loading)\n")
            elif self.strategy:
                logger.info("Applying strategy...")
                actual_model = self.model.model if hasattr(self.model, 'model') else self.model
                self.strategy.apply(actual_model)
                logger.info("✓ Strategy applied\n")
            
            # Create evaluators with model
            self._create_evaluators()
            
            # Run evaluations
            logger.info("Running evaluations...")
            logger.info("-"*80 + "\n")
            
            all_results = {}
            for evaluator in self.evaluators:
                logger.info(f"Running {evaluator.__class__.__name__}...")
                results = evaluator.evaluate(model=self.model)  # ← No model parameter
                all_results[evaluator.__class__.__name__] = results
                logger.info(f"✓ {evaluator.__class__.__name__} complete\n")
            
            # Save results
            self.save_results(all_results)
            
            logger.info("="*80)
            logger.info("✓ EXPERIMENT COMPLETE")
            logger.info("="*80)
            
            return all_results
        
        except Exception as e:
            logger.error(f"✗ Experiment failed: {e}")
            raise
    
    def save_results(self, results: dict):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config["experiment"]["name"]
        filename = f"{exp_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        output = {
            "experiment": self.config["experiment"],
            "model": self.config["model"],
            "strategy": self.config.get("strategy"),
            "results": results,
            "timestamp": timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"✓ Results saved to: {filepath}")
