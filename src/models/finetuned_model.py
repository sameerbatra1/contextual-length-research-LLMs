# src/models/finetuned_model.py
"""
Shared loader for base models with a context-extension LoRA adapter.

The RoPE method used at evaluation must be the one used in training. The
trainer writes it to `extension_config.json` beside the adapter; this loader
reads it from there. Adapters trained before that file existed need
`rope_method` given explicitly in the experiment config.
"""
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base_models import BaseModel
from src.rope import EXTENSION_CONFIG_NAME, apply_rope_scaling

logger = logging.getLogger(__name__)
load_dotenv()


def read_extension_config(adapter_path: str):
    """Find the extension config written by the trainer (adapter dir or its parent)."""
    for folder in (Path(adapter_path), Path(adapter_path).parent):
        candidate = folder / EXTENSION_CONFIG_NAME
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


class FinetunedExtendedModel(BaseModel):
    """Base model + LoRA adapter, with the adapter's RoPE scaling applied."""

    DEFAULT_BASE_MODEL = None
    DEFAULT_ADAPTER_PATH = None
    DEFAULT_CONTEXT_LENGTH = 8192
    NATIVE_CONTEXT_LENGTH = 2048
    MODEL_TYPE = "finetuned"

    def __init__(self):
        super().__init__()
        self.model_name = self.MODEL_TYPE
        self.native_context_length = self.NATIVE_CONTEXT_LENGTH
        self.adapter_path = None
        self.extended_context_length = None
        self.rope_info = None

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
        instance = cls()
        instance.load(
            model_path=model_cfg.get("base_model_path", model_cfg.get("base_path")),
            adapter_path=model_cfg.get("adapter_path"),
            context_length=model_cfg.get("context_length"),
            rope_method=model_cfg.get("rope_method"),
        )
        instance.config_dict = config
        return instance

    def load(
        self,
        model_path: str = None,
        adapter_path: str = None,
        context_length: int = None,
        rope_method: str = None,
        **kwargs,
    ):
        model_path = model_path or self.DEFAULT_BASE_MODEL
        adapter_path = adapter_path or self.DEFAULT_ADAPTER_PATH
        if not adapter_path or not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        extension = read_extension_config(adapter_path)
        if extension:
            trained_method = extension["rope_method"]
            if rope_method and rope_method != trained_method:
                raise ValueError(
                    f"Config asks for rope_method='{rope_method}' but the adapter at "
                    f"{adapter_path} was trained with '{trained_method}'."
                )
            rope_method = trained_method
            context_length = context_length or extension["target_length"]
            native_length = extension.get("original_length", self.NATIVE_CONTEXT_LENGTH)
        else:
            if not rope_method:
                raise ValueError(
                    f"No {EXTENSION_CONFIG_NAME} beside the adapter at {adapter_path}, and no "
                    "model.rope_method in the experiment config. State the RoPE method the "
                    "adapter was trained with (none, linear, yarn or dynamic)."
                )
            native_length = self.NATIVE_CONTEXT_LENGTH
        context_length = context_length or self.DEFAULT_CONTEXT_LENGTH

        print(f"Loading {self.MODEL_TYPE}: base={model_path} adapter={adapter_path}")
        token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
            **kwargs,
        )
        self.rope_info = apply_rope_scaling(self.model, rope_method, native_length, context_length)
        print(f"✓ RoPE scaling applied and verified: {self.rope_info}")

        self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model = self.model.merge_and_unload()
        self.adapter_path = adapter_path

        self.device = next(self.model.parameters()).device
        self.model.eval()
        self.tokenizer.model_max_length = context_length
        self.extended_context_length = context_length
        self.config = {
            "model_type": self.MODEL_TYPE,
            "base_model_path": model_path,
            "adapter_path": adapter_path,
            "native_context_length": native_length,
            "extended_context_length": context_length,
            "rope": self.rope_info,
            "extension_config": extension,
            "device": str(self.device),
            "dtype": "bfloat16",
        }
        return self

    def generate(self, prompt: str, max_tokens: int = 100, **generation_kwargs) -> str:
        """Greedy decoding by default, the same as the base-model wrappers."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(generation_kwargs)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)
        return self.tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    def get_model_info(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        return {**self.config, "total_parameters": f"{total:,}"}
