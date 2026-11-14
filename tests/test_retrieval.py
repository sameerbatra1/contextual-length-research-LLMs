"""CPU tests for the Phase 1 retrieval task, using a whitespace tokenizer (no downloads)."""
import pytest

from src.evaluators.retrieval import NEEDLE, RetrievalTask, make_trial


class WordTokenizer:
    """One token per whitespace-separated word; newlines are kept as tokens."""
    bos_token_id = 0
    chat_template = None

    def __init__(self):
        self.vocab, self.words = {"<bos>": 0}, ["<bos>"]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.replace("\n", " \n ").split(" "):
            if word == "":
                continue
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids).replace(" \n ", "\n")


HAYSTACK = " ".join(f"word{i}" for i in range(5000))


@pytest.fixture
def task():
    return RetrievalTask(WordTokenizer(), use_chat_template=False, haystack_text=HAYSTACK)


def text_of(task, prompt):
    return task.tokenizer.decode(prompt["input_ids"])


@pytest.mark.parametrize("kind,distractors", [("single", 0), ("multi", 20)])
@pytest.mark.parametrize("length", [400, 2000, 9000])
def test_prompt_fits_and_holds_every_needle(task, kind, distractors, length):
    for depth in (0.0, 0.5, 1.0):
        trial = make_trial(42, length, depth, 0, kind, distractors)
        prompt = task.build(trial, length, depth)
        assert length - 40 <= prompt["prompt_tokens"] <= length
        text = text_of(task, prompt)
        assert text.count(NEEDLE.format(key=trial["key"], number=trial["number"])) == 1
        assert len(trial["distractors"]) == distractors
        for d in trial["distractors"]:
            assert NEEDLE.format(key=d["key"], number=d["number"]) in text
        assert text.rstrip().endswith(f"The special magic number for {trial['key']} is")


def test_depth_places_the_needle(task):
    positions = []
    for depth in (0.0, 0.5, 1.0):
        trial = make_trial(42, 2000, depth, 0, "single", 0)
        text = text_of(task, task.build(trial, 2000, depth))
        positions.append(text.index(trial["number"]) / len(text))
    assert positions[0] < 0.1 and 0.4 < positions[1] < 0.6 and positions[2] > 0.9


def test_trials_are_reproducible_and_distinct():
    a = make_trial(42, 2000, 0.5, 0, "multi", 20)
    assert a == make_trial(42, 2000, 0.5, 0, "multi", 20)
    assert a["number"] != make_trial(42, 2000, 0.5, 1, "multi", 20)["number"]
    keys = [a["key"]] + [d["key"] for d in a["distractors"]]
    assert len(set(keys)) == 21


def test_scoring():
    trial = make_trial(42, 2000, 0.5, 0, "multi", 3)
    wrong = trial["distractors"][0]["number"]
    assert RetrievalTask.score(f" {trial['number']}.", trial)["correct"]
    assert RetrievalTask.score(f"{int(trial['number']):,}", trial)["correct"]
    assert not RetrievalTask.score("I do not know", trial)["correct"]
    assert RetrievalTask.score("I do not know", trial)["returned_no_number"]
    assert RetrievalTask.score(wrong, trial)["returned_distractor"]
    assert not RetrievalTask.score(f"{trial['number']} or {wrong}", trial)["correct"]
