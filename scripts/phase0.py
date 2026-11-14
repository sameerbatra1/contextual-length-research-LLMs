#!/usr/bin/env python3
"""
Phase 0: re-run the 2k -> 8k extension pilot with verified RoPE scaling.

Question: once scaling is applied correctly and identically in training and
evaluation, does TinyLlama still fail to retrieve at 4k/8k while Phi-2 recovers?

    python scripts/phase0.py plan                      # print every command
    python scripts/phase0.py train --model tinyllama --method linear --seed 42
    python scripts/phase0.py eval  --model tinyllama --condition ft_linear --seed 42
    python scripts/phase0.py eval  --model tinyllama --condition base
    python scripts/phase0.py summary
    python scripts/phase0.py compare --model tinyllama   # paired significance tests

Conditions
    base        unmodified model
    zs_linear   linear PI applied, no fine-tuning
    zs_yarn     YaRN applied, no fine-tuning
    ft_none     LoRA fine-tuned at 8k with unmodified RoPE (control)
    ft_linear   LoRA fine-tuned with linear PI
    ft_yarn     LoRA fine-tuned with YaRN
"""
import argparse
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS = {
    "phi2": {"path": "microsoft/phi-2", "base": "Phi2Model", "finetuned": "Phi2FinetunedModel"},
    "tinyllama": {
        "path": "TinyLlama/TinyLlama_v1.1",
        "base": "tinyLlamaModel",
        "finetuned": "TinyLlamaFinetunedModel",
    },
}
ZERO_SHOT = {"base": None, "zs_linear": "LinearPIStrategy", "zs_yarn": "YaRNStrategy"}
FINETUNED = {"ft_none": "none", "ft_linear": "linear", "ft_yarn": "yarn"}
ORIGINAL, TARGET = 2048, 8192
LENGTHS = [2048, 4096, 8192]
TRAIN_DATA = "data/pile_train_stratified/train_documents.json"
RESULTS_DIR = "results/phase0"


def checkpoint_dir(model, method, seed):
    return f"checkpoints/phase0/{model}_{method}_s{seed}"


def build_config(model, condition, seed, evals):
    spec = MODELS[model]
    name = f"phase0_{model}_{condition}_s{seed}"
    config = {
        "experiment": {"name": name, "description": "Phase 0 validation run", "seed": seed},
        "logging": {"log_dir": "logs/phase0", "log_file": f"{name}.txt"},
    }
    if condition in ZERO_SHOT:
        config["model"] = {"type": spec["base"], "path": spec["path"]}
        strategy = ZERO_SHOT[condition]
        config["strategy"] = (
            {"type": strategy, "original_length": ORIGINAL, "target_length": TARGET}
            if strategy else None
        )
    else:
        config["model"] = {
            "type": spec["finetuned"],
            "base_path": spec["path"],
            "adapter_path": f"{checkpoint_dir(model, FINETUNED[condition], seed)}/final_model",
        }
        config["strategy"] = None

    evaluation = {}
    if "needle" in evals:
        evaluation["needle_haystack"] = {
            "context_lengths": LENGTHS,
            "depths": [round(0.1 * i, 1) for i in range(11)],
            "num_trials": 5,
        }
    if "quality" in evals:
        if os.path.isdir("data/quality"):
            evaluation["quality"] = {"context_lengths": LENGTHS, "num_samples": 100, "scoring": "logits"}
        else:
            print("⚠ data/quality not found: skipping QuALITY (python data/data/getting_quality.py)")
    if "perplexity" in evals:
        if os.path.exists("data/pile_test_stratified/test_documents.json"):
            evaluation["perplexity"] = {"context_lengths": LENGTHS, "num_samples": 50, "stride": 512}
        else:
            print("⚠ data/pile_test_stratified not found: skipping perplexity (python data/data/getting_data.py)")
    config["evaluation"] = evaluation
    return config


def cmd_train(args):
    from src.training.continue_pretrain import ContextExtensionTrainer

    trainer = ContextExtensionTrainer(
        model_name=MODELS[args.model]["path"],
        context_length=TARGET,
        method=args.method,
        original_length=ORIGINAL,
        use_4bit=args.use_4bit,
        seed=args.seed,
    )
    trainer.train(
        data_path=TRAIN_DATA,
        output_dir=checkpoint_dir(args.model, args.method, args.seed),
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
    )


def cmd_eval(args):
    from src.experiment import ExperimentRunner

    config = build_config(args.model, args.condition, args.seed, args.evals.split(","))
    ExperimentRunner(config, output_dir=RESULTS_DIR).run()


def cmd_plan(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    for model in MODELS:
        for condition in ZERO_SHOT:
            for seed in seeds:
                print(f"python scripts/phase0.py eval --model {model} --condition {condition} --seed {seed}")
        for condition, method in FINETUNED.items():
            for seed in seeds:
                print(f"python scripts/phase0.py train --model {model} --method {method} --seed {seed}")
                print(f"python scripts/phase0.py eval --model {model} --condition {condition} --seed {seed}")


def wilson(passes, total, z=1.96):
    """95% Wilson interval for a pass rate."""
    if total == 0:
        return (0.0, 0.0)
    p = passes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cmd_summary(args):
    rows = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/phase0_*.json")):
        with open(path) as f:
            data = json.load(f)
        name = data["experiment"]["name"].replace("phase0_", "")
        key = name.rsplit("_s", 1)[0]  # pool seeds
        row = rows.setdefault(key, {"needle": {}, "quality": {}, "ppl": {}, "files": 0})
        row["files"] += 1
        results = data["results"]
        for detail in results.get("NeedleHaystackEvaluator", {}).get("details", []):
            passed, total = row["needle"].get(detail["context_length"], (0, 0))
            row["needle"][detail["context_length"]] = (
                passed + detail["trials"]["passed"], total + detail["trials"]["total"]
            )
        for length, stats in results.get("QualityEvaluator", {}).get("by_context_length", {}).items():
            passed, total = row["quality"].get(int(length), (0, 0))
            row["quality"][int(length)] = (passed + stats["correct"], total + stats["num_samples"])
        for length, stats in results.get("PerplexityEvaluator", {}).get("summary", {}).items():
            row["ppl"].setdefault(int(length), []).append(stats["average_perplexity"])

    if not rows:
        print(f"No results in {RESULTS_DIR} yet.")
        return
    for metric, label in (("needle", "Needle trial pass rate"), ("quality", "QuALITY accuracy (chance 25%)")):
        print(f"\n{label}, pooled over seeds, with 95% Wilson interval")
        print(f"{'condition':<28}" + "".join(f"{length:>24}" for length in LENGTHS))
        for key, row in rows.items():
            cells = []
            for length in LENGTHS:
                if length in row[metric]:
                    passed, total = row[metric][length]
                    low, high = wilson(passed, total)
                    cells.append(f"{passed / total:6.1%} [{low:4.0%},{high:4.0%}] n={total}")
                else:
                    cells.append("-")
            print(f"{key:<28}" + "".join(f"{c:>24}" for c in cells))
    print("\nPerplexity (mean over seeds)")
    print(f"{'condition':<28}" + "".join(f"{length:>12}" for length in LENGTHS))
    for key, row in rows.items():
        cells = [f"{sum(v) / len(v):.1f}" if (v := row["ppl"].get(length)) else "-" for length in LENGTHS]
        print(f"{key:<28}" + "".join(f"{c:>12}" for c in cells))

# Comparisons fixed before any result was seen. A is the reference, B the treatment.
COMPARISONS = [
    ("base", "zs_linear"), ("base", "zs_yarn"),
    ("base", "ft_none"), ("base", "ft_linear"), ("base", "ft_yarn"),
    ("ft_none", "ft_linear"), ("ft_none", "ft_yarn"),
    ("ft_linear", "ft_yarn"),
]


def load_items(model, condition, seed):
    """Per-item outcomes of one run, keyed so the same item can be matched across runs."""
    # Result files end in a timestamp; a re-run of the same condition supersedes older ones.
    paths = sorted(glob.glob(f"{RESULTS_DIR}/phase0_{model}_{condition}_s{seed}_2*.json"))
    if not paths:
        return None
    with open(paths[-1]) as f:
        results = json.load(f)["results"]
    items = {"needle": {}, "quality": {}, "ppl": {}}
    for d in results.get("NeedleHaystackEvaluator", {}).get("details", []):
        for i, ok in enumerate(d["trials"]["results"]):
            items["needle"][(d["context_length"], d["depth"], i)] = float(bool(ok))
    for d in results.get("QualityEvaluator", {}).get("details", []):
        items["quality"][(d["context_length"], d["article_id"], d["question"])] = float(bool(d["success"]))
    for d in results.get("PerplexityEvaluator", {}).get("detailed_results", []):
        if "avg_loss" in d and math.isfinite(d["avg_loss"]):
            items["ppl"][(d["context_length"], d["doc_id"])] = float(d["avg_loss"])
    return items


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired pass/fail outcomes."""
    only_a = sum(1 for x, y in zip(a, b) if x > y)
    only_b = sum(1 for x, y in zip(a, b) if y > x)
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(a, b, resamples=10000, seed=0):
    """95% interval for mean(b - a), resampling items with their pairing kept."""
    import numpy as np

    diff = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_test(a, b, resamples=10000, seed=0):
    """Two-sided paired permutation test for mean(b - a) = 0."""
    import numpy as np

    diff = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    flipped = (rng.choice([-1.0, 1.0], size=(resamples, len(diff))) * diff).mean(axis=1)
    return float((1 + (abs(flipped) >= abs(diff.mean()) - 1e-12).sum()) / (resamples + 1))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cmd_compare(args):
    runs = {c: load_items(args.model, c, args.seed) for c in [*ZERO_SHOT, *FINETUNED]}
    labels = {
        "needle": "Needle pass rate (exact McNemar test)",
        "quality": "QuALITY accuracy (exact McNemar test)",
        "ppl": "Perplexity, geometric mean over documents (paired permutation test)",
    }
    for metric, label in labels.items():
        rows = []
        for ref, treat in COMPARISONS:
            if not runs[ref] or not runs[treat]:
                continue
            for length in LENGTHS:
                keys = sorted(
                    k for k in runs[ref][metric] if k[0] == length and k in runs[treat][metric]
                )
                if not keys:
                    continue
                a = [runs[ref][metric][k] for k in keys]
                b = [runs[treat][metric][k] for k in keys]
                low, high = paired_bootstrap(a, b)
                p = sign_flip_test(a, b) if metric == "ppl" else mcnemar_exact(a, b)
                rows.append((ref, treat, length, len(keys), sum(a) / len(a), sum(b) / len(b), low, high, p))
        if not rows:
            continue
        adjusted = holm([r[-1] for r in rows])
        print(f"\n{label}: {args.model}, seed {args.seed}")
        print("B minus A with a 95% paired-bootstrap interval. 'p adj' is Holm-corrected "
              f"over the {len(rows)} tests in this table; * means p adj < 0.05.")
        print(f"{'A':<10}{'B':<11}{'length':>7}{'pairs':>7}{'A':>9}{'B':>9}   {'B vs A':<26}{'p':>9}{'p adj':>9}")
        for (ref, treat, length, n, mean_a, mean_b, low, high, p), p_adj in zip(rows, adjusted):
            if metric == "ppl":
                va, vb = f"{math.exp(mean_a):.1f}", f"{math.exp(mean_b):.1f}"
                effect = f"x{math.exp(mean_b - mean_a):.2f} [x{math.exp(low):.2f}, x{math.exp(high):.2f}]"
            else:
                va, vb = f"{mean_a:.1%}", f"{mean_b:.1%}"
                effect = f"{mean_b - mean_a:+.1%} [{low:+.1%}, {high:+.1%}]"
            star = " *" if p_adj < 0.05 else ""
            print(f"{ref:<10}{treat:<11}{length:>7}{n:>7}{va:>9}{vb:>9}   {effect:<26}{p:>9.4f}{p_adj:>9.4f}{star}")
    if not any(runs.values()):
        print(f"No results for {args.model} seed {args.seed} in {RESULTS_DIR}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--model", required=True, choices=MODELS)
    train.add_argument("--method", required=True, choices=sorted(set(FINETUNED.values())))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max_steps", type=int, default=400)
    train.add_argument("--global_batch_size", type=int, default=16)
    train.add_argument("--use_4bit", action="store_true", help="Train in 4-bit (the pilot's setting)")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--model", required=True, choices=MODELS)
    evaluate.add_argument("--condition", required=True, choices=[*ZERO_SHOT, *FINETUNED])
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--evals", default="needle,quality,perplexity")
    evaluate.set_defaults(func=cmd_eval)

    plan = sub.add_parser("plan")
    plan.add_argument("--seeds", default="42")
    plan.set_defaults(func=cmd_plan)

    summary = sub.add_parser("summary")
    summary.set_defaults(func=cmd_summary)

    compare = sub.add_parser("compare", help="Paired significance tests between conditions")
    compare.add_argument("--model", required=True, choices=MODELS)
    compare.add_argument("--seed", type=int, default=42)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
