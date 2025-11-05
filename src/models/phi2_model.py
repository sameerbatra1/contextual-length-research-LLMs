from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.models.base_models import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class Phi2Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "phi2"
        self.native_context_length = 2048
    
    def load(self, model_path: str = "microsoft/phi-2", **kwargs):
        """Load Phi2 model and tokenizer"""
        print(f"Loading Phi2 from {model_path}...")
        
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
            "model_type": "phi2",
            "model_path": model_path,
            "original_max_length": self.model.config.max_position_embeddings,
            "device": str(self.device),
            "dtype": "bfloat16"
        }
        
        print(f"✓ Model loaded on {self.device}")
        return self

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt - returns ONLY the generated part"""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]  # ← Store this BEFORE generation
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
                # NO output_only_new_tokens parameter!
            )
        
        # Manually extract only the NEW tokens using slicing
        generated_tokens = outputs[:, input_length:]  # ← This is the key!
        
        # Decode only the generated part
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return answer
    
    def apply_rope_scaling(self, rope_config: dict):
        """Apply RoPE scaling configuration"""
        self.model.config.rope_scaling = rope_config
        print(f"✓ RoPE scaling applied: {rope_config}")