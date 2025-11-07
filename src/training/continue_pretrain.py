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


class ContextExtensionTrainer:
    """Fine-tune Phi-2 with YaRN-inspired hyperparameters and 4-bit quantization"""
    
    def __init__(self, model_name: str, context_length: int, scale_factor: int = 16):
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
        
        # LoRA config (train only adapters, not full model)
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "v_proj"],  # Which layers to add LoRA
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  # Show how many params are trainable
        
        logger.info(f"✓ Model loaded with 4-bit quantization + LoRA")
        logger.info(f"✓ Linear PI scaling: 2048 → {context_length}")
    
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
        per_device_batch_size = 1  # ← Increased from 1
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
