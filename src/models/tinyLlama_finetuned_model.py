# src/models/tinyLlama_finetuned_model.py
from src.models.finetuned_model import FinetunedExtendedModel


class TinyLlamaFinetunedModel(FinetunedExtendedModel):
    """TinyLlama with a context-extension LoRA adapter."""

    DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama_v1.1"
    MODEL_TYPE = "tinyllama_finetuned"
