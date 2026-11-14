# src/models/phi2_finetuned_model.py
from src.models.finetuned_model import FinetunedExtendedModel


class Phi2FinetunedModel(FinetunedExtendedModel):
    """Phi-2 with a context-extension LoRA adapter."""

    DEFAULT_BASE_MODEL = "microsoft/phi-2"
    MODEL_TYPE = "phi2_finetuned"
