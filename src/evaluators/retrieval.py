"""
Key-value retrieval in a long text, for measuring effective context length.

A sentence "The special magic number for <key> is <number>." is placed at a
chosen depth in a haystack of essays. The model is asked for that number.

Tasks
    single   one needle, no distractors (easy: the needle is the only sentence
             of its kind)
    multi    the same needle plus `num_distractors` sentences of the same form
             with other keys and numbers (the model must match the key)

Every trial is built from a random generator seeded by (seed, length, depth,
trial), so all models are asked about the same key, number, depth and essay
region, whatever their tokenizer.
"""
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

KEYS = [
    "Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn", "Havana", "Perth", "Dakar", "Bergen", "Lima",
    "Tunis", "Hanoi", "Minsk", "Cusco", "Aleppo", "Tbilisi", "Windhoek", "Malmo", "Porto", "Kyoto",
    "Belgrade", "Accra", "Montevideo", "Yerevan", "Reykjavik", "Kathmandu", "Lagos", "Valencia",
    "Gdansk", "Muscat", "Pretoria", "Bogota", "Salzburg", "Jakarta", "Tashkent", "Halifax", "Dublin",
    "Riga", "Karachi", "Mombasa", "Zagreb", "Cordoba", "Bilbao", "Dhaka", "Sofia", "Leipzig",
    "Adelaide", "Fresno", "Oaxaca", "Tirana", "Kigali", "Bruges", "Manila", "Tromso", "Amman",
    "Lusaka", "Ankara", "Geneva", "Darwin", "Seville",
]
NEEDLE = "The special magic number for {key} is {number}."
INSTRUCTION = (
    "Below is a long text. Special magic numbers are hidden inside it. "
    "Read the text, then answer the question that follows it.\n\n"
)
QUESTION = "\n\nQuestion: What is the special magic number for {key}?"
CHAT_SUFFIX = " Answer with the number only."
PLAIN_SUFFIX = "\nAnswer: The special magic number for {key} is"
HAYSTACK_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "PaulGrahamEssays.txt"


def make_trial(seed: int, length: int, depth: float, trial: int, task: str, num_distractors: int) -> Dict[str, Any]:
    """The tokenizer-independent content of one trial."""
    rng = random.Random(f"{seed}|{length}|{depth}|{trial}")
    keys = rng.sample(KEYS, 1 + (num_distractors if task == "multi" else 0))
    numbers = rng.sample(range(1_000_000, 10_000_000), len(keys))
    return {
        "key": keys[0],
        "number": str(numbers[0]),
        "distractors": [
            {"key": k, "number": str(n), "position": rng.random()} for k, n in zip(keys[1:], numbers[1:])
        ],
        "haystack_start": rng.random(),
    }


class RetrievalTask:
    def __init__(self, tokenizer, use_chat_template: bool, haystack_text: Optional[str] = None):
        self.tokenizer = tokenizer
        self.use_chat = use_chat_template
        text = haystack_text if haystack_text is not None else HAYSTACK_PATH.read_text(encoding="utf-8")
        self.haystack: List[int] = tokenizer.encode(text, add_special_tokens=False)
        if len(self.haystack) < 100:
            raise ValueError("Haystack text is too short.")

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _wrap(self, user_text: str, key: str) -> List[int]:
        """Token ids of the full prompt, in the model's own chat format if it has one."""
        if self.use_chat:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text + CHAT_SUFFIX}],
                add_generation_prompt=True, tokenize=True,
            )
            return list(ids["input_ids"] if isinstance(ids, dict) else ids)
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        return bos + self._encode(user_text + PLAIN_SUFFIX.format(key=key))

    def build(self, trial: Dict[str, Any], length: int, depth: float) -> Dict[str, Any]:
        """Build a prompt of at most `length` tokens with the needle at `depth` (0 start, 1 end)."""
        question = QUESTION.format(key=trial["key"])
        overhead = len(self._wrap(INSTRUCTION + question, trial["key"]))
        needle = self._encode("\n" + NEEDLE.format(key=trial["key"], number=trial["number"]) + "\n")
        extras = [
            (d["position"], self._encode("\n" + NEEDLE.format(key=d["key"], number=d["number"]) + "\n"))
            for d in trial["distractors"]
        ]
        # 8 tokens of slack: decoding and re-encoding a cut haystack can shift the count slightly
        body_budget = length - overhead - len(needle) - sum(len(e) for _, e in extras) - 8
        if body_budget < 50:
            raise ValueError(f"Length {length} is too short for this task.")

        start = int(trial["haystack_start"] * len(self.haystack))
        body = (self.haystack[start:] + self.haystack[:start])
        while len(body) < body_budget:
            body = body + body
        body = body[:body_budget]

        inserts = sorted([(depth, needle)] + extras, key=lambda item: item[0])
        context, cursor = [], 0
        for position, tokens in inserts:
            cut = int(position * body_budget)
            context += body[cursor:cut] + tokens
            cursor = cut
        context += body[cursor:]

        user_text = INSTRUCTION + self.tokenizer.decode(context, skip_special_tokens=True) + question
        input_ids = self._wrap(user_text, trial["key"])
        return {"input_ids": input_ids, "prompt_tokens": len(input_ids)}

    @staticmethod
    def score(answer: str, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Correct if the target number is in the answer and no distractor number is."""
        found = set(re.findall(r"\d+", answer.replace(",", "")))
        distractor_hit = any(d["number"] in found for d in trial["distractors"])
        return {
            "correct": trial["number"] in found and not distractor_hit,
            "returned_distractor": distractor_hit,
            "returned_no_number": not found,
        }
