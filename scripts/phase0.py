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

    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
