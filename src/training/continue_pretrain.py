# src/training/continue_pretrain.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patch_phi_rope():
    """Monkey-patch Phi's RoPE implementation to fix device mismatch"""
    try:
        from transformers.models.phi.modeling_phi import PhiRotaryEmbedding
        
        # Store original _dynamic_frequency_update if it exists
        _original_dynamic_freq_update = PhiRotaryEmbedding._dynamic_frequency_update if hasattr(PhiRotaryEmbedding, '_dynamic_frequency_update') else None
        
        @torch.no_grad()
        def patched_forward(self, x, position_ids):
            """Completely rewritten forward that ensures all tensors are on the same device"""
            device = x.device
            
            # Move position_ids to correct device
            if isinstance(position_ids, torch.Tensor) and position_ids.device != device:
                position_ids = position_ids.to(device)
            
            # Handle dynamic frequency update
            if "dynamic" in self.rope_type:
                if _original_dynamic_freq_update is not None:
                    _original_dynamic_freq_update(self, position_ids, device=device)
                # Ensure inv_freq is on the correct device after dynamic update
                if hasattr(self, 'inv_freq') and isinstance(self.inv_freq, torch.Tensor):
                    if self.inv_freq.device != device:
                        self.inv_freq = self.inv_freq.to(device)
            
            # Ensure inv_freq is on the correct device
            if hasattr(self, 'inv_freq') and isinstance(self.inv_freq, torch.Tensor):
                if self.inv_freq.device != device:
                    self.inv_freq = self.inv_freq.to(device)
            
            # Core RoPE block - now everything is on the same device
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            
            # Force float32
            device_type = device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            
            # Apply attention scaling
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling
            
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
        # Replace the forward method
        PhiRotaryEmbedding.forward = patched_forward
        logger.info("✓ Monkey-patched PhiRotaryEmbedding.forward to fix device mismatch")
        return True
    except Exception as e:
        logger.warning(f"Could not monkey-patch Phi RoPE: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def _patch_llama_rope():
    """Monkey-patch Llama's RoPE implementation to fix device mismatch"""
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        
        @torch.no_grad()
        def patched_forward(self, x, position_ids):
            """Completely rewritten forward that ensures all tensors are on the same device"""
            device = x.device
            
            # Move position_ids to correct device
            if isinstance(position_ids, torch.Tensor) and position_ids.device != device:
                position_ids = position_ids.to(device)
            
            # Ensure inv_freq is on the correct device
            if hasattr(self, 'inv_freq') and isinstance(self.inv_freq, torch.Tensor):
                if self.inv_freq.device != device:
                    self.inv_freq = self.inv_freq.to(device)
            
            # Core RoPE block - now everything is on the same device
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            
            # Force float32
            device_type = device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
        # Replace the forward method
        LlamaRotaryEmbedding.forward = patched_forward
        logger.info("✓ Monkey-patched LlamaRotaryEmbedding.forward to fix device mismatch")
        return True
    except Exception as e:
        logger.warning(f"Could not monkey-patch Llama RoPE: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def _patch_gpt_neox_rope():
    """Monkey-patch GPT-NeoX's RoPE implementation to fix device mismatch"""
    try:
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
        
        @torch.no_grad()
        def patched_forward(self, x, position_ids):
            """Completely rewritten forward that ensures all tensors are on the same device"""
            device = x.device
            
            # Move position_ids to correct device
            if isinstance(position_ids, torch.Tensor) and position_ids.device != device:
                position_ids = position_ids.to(device)
            
            # Ensure inv_freq is on the correct device
            if hasattr(self, 'inv_freq') and isinstance(self.inv_freq, torch.Tensor):
                if self.inv_freq.device != device:
                    self.inv_freq = self.inv_freq.to(device)
            
            # Core RoPE block - now everything is on the same device
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            
            # Force float32
            device_type = device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
        # Replace the forward method
        GPTNeoXRotaryEmbedding.forward = patched_forward
        logger.info("✓ Monkey-patched GPTNeoXRotaryEmbedding.forward to fix device mismatch")
        return True
    except Exception as e:
        logger.warning(f"Could not monkey-patch GPT-NeoX RoPE: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# Apply the patches before any model loading
_patch_phi_rope()
_patch_llama_rope()
_patch_gpt_neox_rope()


class ContextExtensionTrainer:
    """Fine-tune Phi-2 with YaRN-inspired hyperparameters and 4-bit quantization"""
    
    def __init__(self, model_name: str, context_length: int, scale_factor: int = 4):
        self.model_name = model_name
        self.context_length = context_length
        self.scale_factor = scale_factor
        
        logger.info(f"Loading model: {model_name}")
        
        hf_token = os.getenv("HF_TOKEN")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
            bnb_4bit_quant_type="nf4",  # NormalFloat4 (best for LLMs)
            bnb_4bit_compute_dtype=torch.bfloat16  # Faster compute
        )
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,  # ← 4-bit quantization
            device_map="auto",
            token=hf_token,
            rope_scaling={
                "type": "dynamic",
                "factor": context_length / 2048
            }
        )
        
        # Prepare model for k-bit training (required for QLoRA)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Determine target modules based on model architecture
        target_modules = self._get_target_modules(model_name)
        
        # LoRA config (train only adapters, not full model)
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=target_modules,  # Which layers to add LoRA
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  # Show how many params are trainable
        
        logger.info(f"✓ Model loaded with 4-bit quantization + LoRA")
        logger.info(f"✓ Linear PI scaling: 2048 → {context_length}")
    
    def _get_target_modules(self, model_name: str):
        """Determine LoRA target modules based on model architecture"""
        model_name_lower = model_name.lower()
        
        if "pythia" in model_name_lower or "gpt-neox" in model_name_lower:
            # Pythia/GPT-NeoX uses query_key_value projection
            return ["query_key_value"]
        elif "phi" in model_name_lower:
            # Phi models use q_proj, v_proj naming
            return ["q_proj", "v_proj"]
        elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
            # Llama models use q_proj, v_proj naming
            return ["q_proj", "v_proj"]
        else:
            # Default to common naming (Llama-style)
            logger.warning(f"Unknown model architecture for {model_name}, using default target modules")
            return ["q_proj", "v_proj"]
    
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
        """Fine-tune model following Linear PI paper setup"""
        
        logger.info("Loading PILE dataset (streaming)...")
        dataset = load_dataset(
            "json",
            data_files=data_path,
            streaming=True,
            split="train"
        )
        
        logger.info("Preprocessing dataset...")
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # With 4-bit, you can use larger batch size!
        per_device_batch_size = 1 # ← Increased from 1
        gradient_accumulation_steps = 16  # ← Decreased from 16 (still = 16 global)
        
        logger.info(f"Global batch size: {per_device_batch_size * gradient_accumulation_steps}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-5,
            weight_decay=0.0,
            warmup_steps=20,
            max_steps=400,  # Back to 400 steps (as per paper)
            logging_steps=50,
            save_steps=100,
            save_total_limit=3,
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            report_to=[],
            optim="paged_adamw_8bit",  # ← Paged optimizer for QLoRA
            adam_beta1=0.9,
            adam_beta2=0.95,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        logger.info("=" * 70)
        logger.info("Starting Training (4-bit QLoRA)")
        logger.info("=" * 70)
        logger.info(f"Context length: {self.context_length}")
        logger.info(f"Scale factor: {self.scale_factor}x")
        logger.info(f"Steps: 400")
        logger.info(f"Memory mode: 4-bit + LoRA")
        logger.info("=" * 70)
        
        trainer.train()
        
        # Save LoRA adapters (not full model)
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        logger.info(f"✓ Training complete. LoRA adapters saved to {output_dir}/final_model")
