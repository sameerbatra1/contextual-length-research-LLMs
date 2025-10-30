from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.models.base_models import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class GemmaModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "gemma"
        self.native_context_length = 8192
    
    def load(self, model_path: str = "google/gemma-7b", **kwargs):
        """Load Gemma model and tokenizer"""
        print(f"Loading Gemma from {model_path}...")
        
        # Get token from environment
        token = os.getenv("HF_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
            **kwargs
        )
        
        self.device = next(self.model.parameters()).device
        
        self.config = {
            "model_type": "gemma",
            "model_path": model_path,
            "original_max_length": self.model.config.max_position_embeddings,
            "device": str(self.device),
            "dtype": "bfloat16"
        }
        
        print(f"✓ Model loaded on {self.device}")
        return self

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False  # Deterministic for reproducibility
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def apply_rope_scaling(self, rope_config: dict):
        """Apply RoPE scaling configuration"""
        self.model.config.rope_scaling = rope_config
        print(f"✓ RoPE scaling applied: {rope_config}")