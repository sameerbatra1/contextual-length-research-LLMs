#!/usr/bin/env python3
"""
Phase 1, part A: how much of its claimed context can a small model use?

No training. Each model answers key-value retrieval questions at growing
lengths, up to the context length its authors claim.

    python scripts/phase1.py plan
    python scripts/phase1.py eval --model qwen2.5-1.5b --task single
    python scripts/phase1.py eval --model llama3.2-3b --task multi --trials 20
    python scripts/phase1.py eval --path some/hf-model --claimed 32768 --task single
    python scripts/phase1.py summary
    python scripts/phase1.py depth --model phi4-mini --task multi

A length is skipped if it exceeds the model's claimed context. If the GPU runs
out of memory at a length, that length and all longer ones are recorded as
"oom" and are not counted as failures.
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phase0 import wilson  # noqa: E402  (same folder)

# name -> Hugging Face id and the context length claimed on the model card
MODELS = {
    "qwen2.5-1.5b": {"path": "Qwen/Qwen2.5-1.5B-Instruct", "claimed": 32768},
    "qwen2.5-3b": {"path": "Qwen/Qwen2.5-3B-Instruct", "claimed": 32768},
    "llama3.2-1b": {"path": "meta-llama/Llama-3.2-1B-Instruct", "claimed": 131072},
    "llama3.2-3b": {"path": "meta-llama/Llama-3.2-3B-Instruct", "claimed": 131072},
    "phi4-mini": {"path": "microsoft/Phi-4-mini-instruct", "claimed": 131072},
}
FIRST_PASS = ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-3b", "phi4-mini"]
TASKS = ["single", "multi"]
LENGTHS = [2000, 4000, 8000, 16000, 32000, 64000]
DEPTHS = [round(0.1 * i, 1) for i in range(11)]
RESULTS_DIR = "results/phase1"


def environment():
    import torch
    import transformers

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, cwd=ROOT).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {
        "git_commit": commit or None, "git_dirty": dirty,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def cmd_eval(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluators.retrieval import RetrievalTask, make_trial

    if args.model:
        name, path, claimed = args.model, MODELS[args.model]["path"], MODELS[args.model]["claimed"]
    elif args.path and args.claimed:
        name, path, claimed = args.name or args.path.split("/")[-1].lower(), args.path, args.claimed
    else:
        sys.exit("Give --model, or --path together with --claimed.")
    if args.claimed:
        claimed = args.claimed
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else LENGTHS
    lengths = [n for n in lengths if n <= claimed]

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(path)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16 if cuda else torch.float32, device_map={"": 0} if cuda else None
    )
    model.eval()
    use_chat = tokenizer.chat_template is not None and not args.no_chat_template
    task = RetrievalTask(tokenizer, use_chat)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/phase1_{name}_{args.task}_s{args.seed}_{stamp}.json"
    output = {
        "phase": 1, "model": name, "path": path, "task": args.task, "seed": args.seed,
        "claimed_context": claimed,
        "config_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "rope_scaling": getattr(model.config, "rope_scaling", None),
        "chat_template": use_chat, "num_distractors": args.distractors if args.task == "multi" else 0,
        "depths": DEPTHS, "trials_per_depth": args.trials,
        "environment": environment(), "lengths": {}, "complete": False,
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(output, f, indent=1)

    print(f"{name} ({path}), task {args.task}, claimed context {claimed}, lengths {lengths}")
    out_of_memory = False
    for length in lengths:
        if out_of_memory:
            output["lengths"][str(length)] = {"status": "oom", "trials": []}
            continue
        records, started = [], time.time()
        for depth in DEPTHS:
            passed = 0
            for t in range(args.trials):
                trial = make_trial(args.seed, length, depth, t, args.task, args.distractors)
                prompt = task.build(trial, length, depth)
                input_ids = torch.tensor([prompt["input_ids"]], device=model.device)
                try:
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            max_new_tokens=32, do_sample=False, pad_token_id=pad,
                        )
                except torch.cuda.OutOfMemoryError:
                    out_of_memory = True
                    break
                finally:
                    del input_ids
                    if cuda:
                        torch.cuda.empty_cache()
                answer = tokenizer.decode(generated[0, len(prompt["input_ids"]):], skip_special_tokens=True)
                record = {"depth": depth, "trial": t, "key": trial["key"], "number": trial["number"],
                          "prompt_tokens": prompt["prompt_tokens"], "answer": answer.strip()[:80],
                          **task.score(answer, trial)}
                records.append(record)
                passed += record["correct"]
            if out_of_memory:
                break
            print(f"  {length:>6} tokens, depth {depth:.1f}: {passed}/{args.trials}", flush=True)
        if out_of_memory:
            print(f"  {length:>6} tokens: out of GPU memory, this and longer lengths skipped", flush=True)
            output["lengths"][str(length)] = {"status": "oom", "trials": []}
        else:
            correct = sum(r["correct"] for r in records)
            print(f"  {length:>6} tokens: {correct}/{len(records)} correct, {time.time() - started:.0f}s", flush=True)
            output["lengths"][str(length)] = {
                "status": "ok", "seconds": round(time.time() - started), "trials": records
            }
        save()  # after every length, so a crash keeps what is done
    output["complete"] = True
    save()
    print(f"Saved {out_path}")


def load_latest():
    """Newest complete result file per (model, task, seed)."""
    runs = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/phase1_*.json")):
        with open(path) as f:
            data = json.load(f)
        if data.get("complete"):
            runs[(data["model"], data["task"], data["seed"])] = data
    return runs


def cmd_summary(args):
    runs = load_latest()
    if not runs:
        print(f"No complete results in {RESULTS_DIR} yet.")
        return
    lengths = sorted({int(n) for data in runs.values() for n in data["lengths"]})
    for task in TASKS:
        rows = {}
        for (model, run_task, _), data in runs.items():
            if run_task != task:
                continue
            row = rows.setdefault(model, {"claimed": data["claimed_context"], "cells": {}})
            for n, block in data["lengths"].items():
                passed, total, oom = row["cells"].get(int(n), (0, 0, False))
                if block["status"] != "ok":
                    row["cells"][int(n)] = (passed, total, True)
                else:
                    row["cells"][int(n)] = (
                        passed + sum(r["correct"] for r in block["trials"]), total + len(block["trials"]), oom
                    )
        if not rows:
            continue
        print(f"\nTask '{task}': retrieval accuracy with 95% Wilson interval, pooled over seeds")
        print(f"{'model':<14}{'claimed':>8}" + "".join(f"{n:>21}" for n in lengths))
        for model, row in sorted(rows.items()):
            cells = []
            for n in lengths:
                passed, total, oom = row["cells"].get(n, (0, 0, False))
                if total:
                    low, high = wilson(passed, total)
                    cells.append(f"{passed / total:4.0%} [{low:3.0%},{high:4.0%}] n={total}")
                else:
                    cells.append("out of memory" if oom else "-")
            print(f"{model:<14}{row['claimed']:>8}" + "".join(f"{c:>21}" for c in cells))


def cmd_depth(args):
    runs = load_latest()
    data = runs.get((args.model, args.task, args.seed))
    if not data:
        print(f"No complete result for {args.model}, task {args.task}, seed {args.seed}.")
        return
    lengths = [n for n, block in data["lengths"].items() if block["status"] == "ok"]
    print(f"{args.model}, task '{args.task}', seed {args.seed}: correct out of "
          f"{data['trials_per_depth']} by needle depth (0.0 = start of text, 1.0 = just before the question)")
    print(f"{'depth':<8}" + "".join(f"{n:>8}" for n in lengths))
    for depth in data["depths"]:
        cells = [sum(r["correct"] for r in data["lengths"][n]["trials"] if r["depth"] == depth) for n in lengths]
        print(f"{depth:<8}" + "".join(f"{c:>8}" for c in cells))
    if args.task == "multi":
        cells = [sum(r["returned_distractor"] for r in data["lengths"][n]["trials"]) for n in lengths]
        print(f"{'wrong key':<8}" + "".join(f"{c:>8}" for c in cells) + "   (answered with another key's number)")


def cmd_plan(args):
    for model in FIRST_PASS:
        for task in TASKS:
            print(f"atlas run --job p1-{model}-{task} -- python scripts/phase1.py eval --model {model} --task {task}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--model", choices=MODELS)
    evaluate.add_argument("--path", help="Any Hugging Face causal LM, in place of --model")
    evaluate.add_argument("--name", help="Short name for --path models")
    evaluate.add_argument("--claimed", type=int, help="Claimed context length (required with --path)")
    evaluate.add_argument("--task", required=True, choices=TASKS)
    evaluate.add_argument("--trials", type=int, default=10, help="Trials per depth (11 depths)")
    evaluate.add_argument("--distractors", type=int, default=20, help="Other keys in the 'multi' task")
    evaluate.add_argument("--lengths", help="Comma-separated lengths, default " + ",".join(map(str, LENGTHS)))
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--no_chat_template", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    sub.add_parser("plan").set_defaults(func=cmd_plan)
    sub.add_parser("summary").set_defaults(func=cmd_summary)
    depth = sub.add_parser("depth")
    depth.add_argument("--model", required=True)
    depth.add_argument("--task", required=True, choices=TASKS)
    depth.add_argument("--seed", type=int, default=42)
    depth.set_defaults(func=cmd_depth)

    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
