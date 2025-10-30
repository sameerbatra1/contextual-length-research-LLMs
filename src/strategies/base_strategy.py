# src/strategies/base_strategy.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Abstract base class for context extension strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    @abstractmethod
    def apply(self, model):
        """Apply strategy to model"""
        pass
    
    def get_config(self) -> dict:
        """Return strategy configuration for logging"""
        return {
            "name": self.name,
            **self.config
        }
