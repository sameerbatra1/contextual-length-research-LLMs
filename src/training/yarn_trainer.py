# src/training/yarn_trainer.py
"""
YaRN fine-tuning (Peng et al., ICLR 2024).

Thin wrapper over ContextExtensionTrainer with method="yarn". The YaRN
frequencies and attention temperature come from transformers' own
implementation, applied and verified by `src.rope.apply_rope_scaling`.
"""
from src.training.continue_pretrain import ContextExtensionTrainer


class YaRNTrainer(ContextExtensionTrainer):
    def __init__(
        self,
        model_name: str,
        context_length: int,
        alpha: float = 1.0,
        beta: float = 32.0,
        use_dynamic_scaling: bool = False,
        use_4bit: bool = True,
        seed: int = 42,
    ):
        if (alpha, beta) != (1.0, 32.0) or use_dynamic_scaling:
            raise ValueError(
                "Only the paper defaults (alpha=1, beta=32, static scaling) are supported."
            )
        super().__init__(
            model_name=model_name,
            context_length=context_length,
            method="yarn",
            use_4bit=use_4bit,
            seed=seed,
        )
