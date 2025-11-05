# src/training/continue_pretrain.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling  # ← Add this import
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextExtensionTrainer:
    """Fine-tune LLaMA with YaRN-inspired hyperparameters"""
    
    def __init__(self, model_name: str, context_length: int, scale_factor: int = 16):
        self.model_name = model_name
        self.context_length = context_length
        self.scale_factor = scale_factor
        
        logger.info(f"Loading model: {model_name}")
        
        hf_token = os.getenv("HF_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            rope_scaling={
                "type": "dynamic",
                "factor": context_length / 4096
            }
        )
        
        logger.info(f"✓ Model loaded with YaRN scaling: 4096 → {context_length}")
    
    def preprocess_function(self, examples):
        """Tokenize with extended context"""
        tokenized_inputs = self.tokenizer(
            examples["text"],
            max_length=self.context_length,
            truncation=True,
            return_overflowing_tokens=False,  # ← Changed to False
            padding="max_length",
            return_attention_mask=True,
        )
        
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        
        return tokenized_inputs
    
    def train(self, data_path: str, output_dir: str):
        """Fine-tune model following YaRN paper setup"""
        
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
        
        per_device_batch_size = 2
        gradient_accumulation_steps = 32
        
        logger.info(f"Global batch size: {per_device_batch_size * gradient_accumulation_steps}")
        
        # Create data collator with padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Not using masked language modeling
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-5,
            weight_decay=0.0,
            warmup_steps=20,
            max_steps=400,
            logging_steps=50,
            save_steps=100,
            save_total_limit=3,
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            report_to=[],
            optim="adamw_8bit",
            adam_beta1=0.9,
            adam_beta2=0.95,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,  # ← Add this
        )
        
        logger.info("Starting training...")
        logger.info(f"Context length: {self.context_length}")
        logger.info(f"Scale factor: {self.scale_factor}x")
        
        trainer.train()
        
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        logger.info(f"✓ Training complete. Model saved to {output_dir}/final_model")

    # # Phase 2: s=32 (continue from s=16 checkpoint)
    # logger.info("\n" + "=" * 70)
    # logger.info("PHASE 2: Training s=32 (4096 → 131072 tokens) - 200 additional steps")
    # logger.info("=" * 70)
    
    # trainer_s32 = ContextExtensionTrainer(
    #     model_name=f"{output_dir}/phase1_s16/checkpoint_s16",
    #     context_length=131072,
    #     scale_factor=32
    # )
    
    # # Modify training args for phase 2
    # training_args = TrainingArguments(
    #     output_dir=f"{output_dir}/phase2_s32",
    #     num_train_epochs=1,
    #     per_device_train_batch_size=8,
    #     gradient_accumulation_steps=8,
    #     learning_rate=2e-5,
    #     weight_decay=0.0,
    #     warmup_steps=20,
    #     max_steps=200,  # ← Only 200 steps for phase 2
    #     # ... rest same as phase 1
    # )
    
    # logger.info("✓ Complete fine-tuning finished!")
    # logger.info(f"Final model: {output_dir}/phase2_s32/checkpoint_s32")

