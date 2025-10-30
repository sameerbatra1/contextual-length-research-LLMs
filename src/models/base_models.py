# src/models/base_model.py
from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    """Abstract base class for all language models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = {}
    
    @abstractmethod
    def load(self, model_path: str, **kwargs):
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt"""
        pass
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get_config(self) -> dict:
        """Return model configuration for logging"""
        return self.config
    
    def to(self, device: str):
        """Move model to device"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
