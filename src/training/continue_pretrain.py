# src/training/continue_pretrain.py
"""
LoRA fine-tuning for context extension.

One trainer for every RoPE method, so methods are compared under the same
budget (steps, batch size, data order, seed). The RoPE scaling is applied
through `src.rope.apply_rope_scaling`, which verifies it took effect, and the
method is written to `extension_config.json` so evaluation loads the same
position encoding.
"""
import json
import logging
import os
import random

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from src.rope import EXTENSION_CONFIG_NAME, apply_rope_scaling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContextExtensionTrainer:
    """Fine-tune a causal LM with LoRA under a chosen RoPE scaling method."""

    def __init__(
        self,
        model_name: str,
        context_length: int,
        method: str = "linear",
        original_length: int = 2048,
        use_4bit: bool = True,
        seed: int = 42,
        lora_r: int = 8,
        lora_alpha: int = 32,
        scale_factor: int = None,  # unused; accepted for older scripts
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.method = method
        self.original_length = original_length
        self.use_4bit = use_4bit
        self.seed = seed
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scale_factor = context_length / original_length

        set_seed(seed)
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto", token=hf_token
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
            )

        self.rope_info = apply_rope_scaling(
            self.model, method, original_length, context_length
        )
        logger.info(f"✓ RoPE scaling applied and verified: {self.rope_info}")

        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.target_modules = self._get_target_modules(model_name)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        if not use_4bit:
            # Gradient checkpointing needs an input that requires grad when only
            # the LoRA weights are trainable (the 4-bit path does this already).
            self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()

    def _get_target_modules(self, model_name: str):
        """LoRA target modules by architecture."""
        name = model_name.lower()
        if "pythia" in name or "gpt-neox" in name:
            return ["query_key_value"]
        return ["q_proj", "v_proj"]

    def preprocess_function(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            max_length=self.context_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(
        self,
        data_path: str,
        output_dir: str,
        max_steps: int = 400,
        global_batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 20,
    ):
        dataset = load_dataset("json", data_files=data_path, streaming=True, split="train")
        # The training file is grouped by length bin; shuffle so every batch mixes lengths.
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)
        dataset = dataset.map(
            self.preprocess_function, batched=True, remove_columns=dataset.column_names
        )

        per_device_batch_size = 1
        gradient_accumulation_steps = global_batch_size // per_device_batch_size

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.0,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            logging_steps=25,
            save_steps=100,
            save_total_limit=3,
            seed=self.seed,
            data_seed=self.seed,
            ddp_find_unused_parameters=False,
            report_to=[],
        )

        extension_config = {
            "rope_method": self.method,
            "original_length": self.original_length,
            "target_length": self.context_length,
            "scaling_factor": self.scale_factor,
            "rope": self.rope_info,
            "base_model": self.model_name,
            "trained_in_4bit": self.use_4bit,
            "lora": {"r": self.lora_r, "alpha": self.lora_alpha, "target_modules": self.target_modules},
            "max_steps": max_steps,
            "global_batch_size": global_batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "seed": self.seed,
            "data_path": data_path,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
        logger.info("Training configuration:\n" + json.dumps(extension_config, indent=2))

        os.makedirs(output_dir, exist_ok=True)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )
        result = trainer.train()
        extension_config["train_loss"] = result.training_loss

        final_dir = f"{output_dir}/final_model"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        for folder in (output_dir, final_dir):
            with open(os.path.join(folder, EXTENSION_CONFIG_NAME), "w") as f:
                json.dump(extension_config, f, indent=2)
        logger.info(f"✓ Adapter and {EXTENSION_CONFIG_NAME} saved to {final_dir}")
        return extension_config
