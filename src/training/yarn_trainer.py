"""
YaRN Fine-tuning Trainer
Implementation following ICLR 2024 paper specifications

Paper: "YaRN: Efficient Context Window Extension of Large Language Models"
Peng et al., 2024
"""

import torch
import os
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

from src.rope import replace_phi_rope_with_yarn, compute_mscale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YaRNTrainer:
    """
    Fine-tune models with YaRN context extension using exact paper specifications.
    
    Key specifications from paper:
    - NTK-by-parts interpolation (α=1, β=32)
    - Attention temperature scaling (mscale)
    - Training: 400 steps for s=16, 200 additional for s=32
    - Learning rate: 2e-5, no weight decay
    - Warmup: 20 steps (linear)
    - Optimizer: AdamW (β₁=0.9, β₂=0.95)
    - Batch size: 64 (global)
    """
    
    def __init__(
        self, 
        model_name: str, 
        context_length: int,
        alpha: float = 1.0,
        beta: float = 32.0,
        use_dynamic_scaling: bool = False,
        use_4bit: bool = True,
    ):
        """
        Initialize YaRN trainer with paper-specified parameters.
        
        Args:
            model_name: HuggingFace model name (e.g., "microsoft/phi-2")
            context_length: Target context length (e.g., 8192, 16384, 32768)
            alpha: YaRN alpha parameter (default: 1.0, from paper)
            beta: YaRN beta parameter (default: 32.0, from paper)
            use_dynamic_scaling: Enable dynamic scaling at inference (default: False)
            use_4bit: Use 4-bit quantization for memory efficiency (default: True)
        """
        self.model_name = model_name
        self.context_length = context_length
        self.alpha = alpha
        self.beta = beta
        self.use_dynamic_scaling = use_dynamic_scaling
        self.use_4bit = use_4bit
        
        # Compute scaling factor (assuming original context = 2048)
        self.original_length = 2048
        self.scaling_factor = context_length / self.original_length
        
        # Compute mscale (attention temperature)
        self.mscale = compute_mscale(self.scaling_factor)
        
        logger.info("="*70)
        logger.info("YaRN Trainer Initialization")
        logger.info("="*70)
        logger.info(f"Model: {model_name}")
        logger.info(f"Original context: {self.original_length}")
        logger.info(f"Target context: {context_length}")
        logger.info(f"Scaling factor (s): {self.scaling_factor:.2f}x")
        logger.info(f"Alpha (α): {alpha}")
        logger.info(f"Beta (β): {beta}")
        logger.info(f"mscale (sqrt(1/t)): {self.mscale:.4f}")
        logger.info(f"Dynamic scaling: {use_dynamic_scaling}")
        logger.info("="*70)
        
        # Load tokenizer
        logger.info(f"\nLoading tokenizer...")
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model...")
        if use_4bit:
            # 4-bit quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
        
        # Replace RoPE with YaRN
        logger.info(f"\nReplacing RoPE embeddings with YaRN...")
        self.model, num_replaced = replace_phi_rope_with_yarn(
            self.model,
            scaling_factor=self.scaling_factor,
            alpha=alpha,
            beta=beta,
            use_dynamic_scaling=use_dynamic_scaling,
        )
        logger.info(f"✓ Replaced {num_replaced} RoPE layers with YaRN")
        
        # Apply LoRA for parameter-efficient fine-tuning
        logger.info(f"\nApplying LoRA adapters...")
        target_modules = self._get_target_modules(model_name)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("\n" + "="*70)
        logger.info("✓ Model loaded and YaRN applied successfully")
        logger.info("="*70)
    
    def _get_target_modules(self, model_name: str):
        """Determine LoRA target modules based on model architecture"""
        model_name_lower = model_name.lower()
        
        if "phi" in model_name_lower:
            return ["q_proj", "v_proj"]
        elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
            return ["q_proj", "v_proj"]
        elif "pythia" in model_name_lower or "gpt-neox" in model_name_lower:
            return ["query_key_value"]
        else:
            logger.warning(f"Unknown model architecture, using default targets")
            return ["q_proj", "v_proj"]
    
    def _compute_training_steps(self):
        """
        Compute training steps based on paper specifications.
        
        From paper:
        - s=16 (32k context): 400 steps
        - s=32 (64k context): 600 steps (400 + 200)
        """
        base_steps = 400
        
        if self.scaling_factor <= 16:
            return base_steps
        else:
            # Add 200 steps for each doubling beyond s=16
            extra_doublings = int(np.log2(self.scaling_factor / 16))
            return base_steps + (extra_doublings * 200)
    
    def preprocess_function(self, examples):
        """Tokenize with extended context"""
        tokenized_inputs = self.tokenizer(
            examples["text"],
            max_length=self.context_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        
        return tokenized_inputs
    
    def train(self, data_path: str, output_dir: str):
        """
        Fine-tune model following YaRN paper specifications.
        
        Training hyperparameters from paper:
        - Learning rate: 2e-5 (no weight decay)
        - Warmup: 20 steps (linear)
        - Optimizer: AdamW (β₁=0.9, β₂=0.95)
        - Batch size: 64 (global)
        - Steps: 400 for s=16, 600 for s=32
        - Sequence length: Extended context length
        """
        
        # Load dataset
        logger.info("\n" + "="*70)
        logger.info("Loading dataset...")
        logger.info("="*70)
        
        dataset = load_dataset(
            "json",
            data_files=data_path,
            streaming=True,
            split="train"
        )
        
        # Preprocess
        logger.info("Preprocessing dataset...")
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Compute training steps (from paper)
        max_steps = self._compute_training_steps()
        
        # Paper specifications: global batch size = 64
        # Adjust per_device_batch_size and gradient_accumulation_steps
        per_device_batch_size = 1 if self.use_4bit else 2
        gradient_accumulation_steps = 64 // per_device_batch_size
        
        logger.info(f"\nTraining configuration:")
        logger.info(f"  Per-device batch size: {per_device_batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Global batch size: {per_device_batch_size * gradient_accumulation_steps}")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Context length: {self.context_length}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments following paper specifications
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Paper specifications
            learning_rate=2e-5,              # Exact from paper
            weight_decay=0.0,                 # No weight decay (paper)
            warmup_steps=20,                  # 20 steps warmup (paper)
            max_steps=max_steps,              # 400 for s=16, 600 for s=32 (paper)
            
            # Optimizer (AdamW with β₁=0.9, β₂=0.95 from paper)
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            
            # Memory optimizations
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            
            # Logging and saving
            logging_steps=50,
            save_steps=100,
            save_total_limit=3,
            
            # Misc
            ddp_find_unused_parameters=False,
            report_to=[],
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Print training summary
        logger.info("\n" + "="*70)
        logger.info("Starting YaRN Fine-tuning")
        logger.info("="*70)
        logger.info(f"Method: YaRN (NTK-by-parts interpolation)")
        logger.info(f"Context: {self.original_length} → {self.context_length}")
        logger.info(f"Scale: {self.scaling_factor:.2f}x")
        logger.info(f"Alpha (α): {self.alpha}")
        logger.info(f"Beta (β): {self.beta}")
        logger.info(f"mscale: {self.mscale:.4f}")
        logger.info(f"Steps: {max_steps}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"Batch size: {per_device_batch_size * gradient_accumulation_steps}")
        logger.info(f"Warmup: {training_args.warmup_steps} steps")
        logger.info("="*70 + "\n")
        
        # Train
        trainer.train()
        
        # Save LoRA adapters
        logger.info("\n" + "="*70)
        logger.info("Saving model...")
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        
        # Save YaRN configuration
        import json
        yarn_config = {
            "method": "YaRN",
            "original_length": self.original_length,
            "target_length": self.context_length,
            "scaling_factor": self.scaling_factor,
            "alpha": self.alpha,
            "beta": self.beta,
            "mscale": self.mscale,
            "use_dynamic_scaling": self.use_dynamic_scaling,
            "training_steps": max_steps,
            "learning_rate": training_args.learning_rate,
            "global_batch_size": per_device_batch_size * gradient_accumulation_steps,
        }
        
        with open(f"{output_dir}/yarn_config.json", 'w') as f:
            json.dump(yarn_config, f, indent=2)
        
        logger.info(f"✓ LoRA adapters saved to {output_dir}/final_model")
        logger.info(f"✓ YaRN config saved to {output_dir}/yarn_config.json")
        logger.info("="*70)
        logger.info("✅ YaRN fine-tuning complete!")
        logger.info("="*70 + "\n")

