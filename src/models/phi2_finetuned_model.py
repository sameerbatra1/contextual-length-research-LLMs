# src/models/phi2_finetuned_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from src.models.base_models import BaseModel
import os
from dotenv import load_dotenv
import logging
import yaml

logger = logging.getLogger(__name__)
load_dotenv()


class Phi2FinetunedModel(BaseModel):
    """Phi-2 model with fine-tuned LoRA adapters for extended context"""
    
    # Default paths
    DEFAULT_BASE_MODEL = "microsoft/phi-2"
    DEFAULT_ADAPTER_PATH = "checkpoints/phi2_yarn_g5/final_model"
    DEFAULT_CONTEXT_LENGTH = 16384
    NATIVE_CONTEXT_LENGTH = 2048
    
    def __init__(self):
        super().__init__()
        self.model_name = "phi2_finetuned"
        self.native_context_length = self.NATIVE_CONTEXT_LENGTH
        self.adapter_path = None
        self.extended_context_length = None
        self.config = None
    
    @classmethod
    def from_config(cls, config_path: str):
        """Load fine-tuned model from YAML config"""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        instance = cls()
        
        # Get rope_scaling_factor from config if available
        rope_scaling_factor = None
        if 'rope_scaling' in config.get('model', {}):
            rope_scaling_factor = config['model']['rope_scaling'].get('factor')
        
        instance.load(
            model_path=config['model'].get('base_model_path', cls.DEFAULT_BASE_MODEL),
            adapter_path=config['model'].get('adapter_path', cls.DEFAULT_ADAPTER_PATH),
            context_length=config['model'].get('context_length', cls.DEFAULT_CONTEXT_LENGTH),
            rope_scaling_factor=rope_scaling_factor,
        )
        
        instance.config_dict = config
        logger.info("✓ Model loaded from config")
        return instance
    
    def load(
        self,
        model_path: str = None,
        adapter_path: str = None,
        context_length: int = None,
        rope_scaling_factor: float = None,
        **kwargs
    ):
        """
        Load fine-tuned Phi2 model with LoRA adapters
        
        IMPORTANT: Uses the same rope_scaling approach as training (HuggingFace's built-in dynamic scaling).
        
        Uses defaults if not specified:
        - model_path: microsoft/phi-2
        - adapter_path: checkpoints/phi2_yarn_g5/final_model
        - context_length: 16384 (8x extension)
        """
        
        # Use defaults if not provided
        model_path = model_path or self.DEFAULT_BASE_MODEL
        adapter_path = adapter_path or self.DEFAULT_ADAPTER_PATH
        context_length = context_length or self.DEFAULT_CONTEXT_LENGTH
        
        print(f"Loading Phi2 Fine-tuned Model")
        print(f"  Base model: {model_path}")
        print(f"  LoRA adapters: {adapter_path}")
        print(f"  Context length: {context_length}")
        
        # Get HF token
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("HF_TOKEN not set. May fail to download gated models.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Calculate rope scaling factor
        if rope_scaling_factor is None:
            rope_scaling_factor = context_length / self.NATIVE_CONTEXT_LENGTH
        
        print(f"Applying RoPE scaling: {self.NATIVE_CONTEXT_LENGTH} → {context_length} (factor={rope_scaling_factor:.1f}x)")
        
        # IMPORTANT: Use the SAME rope_scaling approach as training
        # Training used HuggingFace's built-in "dynamic" rope_scaling
        rope_config = {
            "type": "dynamic",
            "factor": rope_scaling_factor
        }
        
        # Load base model with rope_scaling (same as training)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
            trust_remote_code=True,
            rope_scaling=rope_config,
            **kwargs
        )
        print(f"✓ Applied rope_scaling: {rope_config}")
        
        # Load and merge LoRA adapters
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        print(f"Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False  # Inference mode
        )
        self.adapter_path = adapter_path
        
        # Merge adapters for faster inference
        self.model = self.model.merge_and_unload()
        print(f"✓ LoRA merged into base model")
        
        self.device = next(self.model.parameters()).device
        self.model.eval()
        
        # Store configuration
        self.extended_context_length = context_length
        self.config = {
            "model_type": "phi2_finetuned",
            "base_model_path": model_path,
            "adapter_path": adapter_path,
            "native_context_length": self.NATIVE_CONTEXT_LENGTH,
            "extended_context_length": self.extended_context_length,
            "rope_scaling": {
                "type": "dynamic",
                "factor": rope_scaling_factor,
            },
            "device": str(self.device),
            "dtype": "bfloat16",
            "status": "ready"
        }
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Extended context: {self.extended_context_length} tokens")
        print(f"✓ Fine-tuned model ready for inference\n")
        
        return self
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **generation_kwargs
    ) -> str:
        """Generate text from prompt (returns only generated part)"""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        
        # Merge generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(generation_kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                **gen_kwargs
            )
        
        # Extract only NEW tokens
        generated_tokens = outputs[:, input_length:]
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        return answer
    
    def generate_with_long_context(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> dict:
        """Generate with extended context, return detailed info"""
        
        # Tokenize with context length limit
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.extended_context_length,
            truncation=True
        ).to(self.device)
        
        input_length = inputs['input_ids'].shape[1]
        context_utilization = (input_length / self.extended_context_length) * 100
        
        print(f"Context: {input_length}/{self.extended_context_length} tokens ({context_utilization:.1f}%)")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only NEW tokens
        generated_tokens = outputs[:, input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )
        
        return {
            "generated_text": generated_text,
            "input_length": input_length,
            "max_context": self.extended_context_length,
            "context_utilization": context_utilization
        }
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            **self.config,
            "total_parameters": f"{total_params:,}",
            "trainable_parameters": f"{trainable_params:,}",
            "trainable_ratio": f"{(trainable_params/total_params)*100:.2f}%"
        }
