# src/evaluators/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Abstract base class for evaluation"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.results = {}
    
    @abstractmethod
    def evaluate(self, model) -> Dict[str, float]:
        """Run evaluation and return metrics"""
        pass
    
    def get_results(self) -> Dict[str, float]:
        """Return evaluation results"""
        return self.results
    
    def reset(self):
        """Reset results for new evaluation"""
        self.results = {}
