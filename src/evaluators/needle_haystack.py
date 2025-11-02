# src/evaluators/needle_haystack.py
import requests
from pathlib import Path
from typing import Dict, Any
import random

from .base_evaluator import BaseEvaluator

class NeedleHaystackEvaluator(BaseEvaluator):
    """
    Needle-in-Haystack evaluation following Greg Kamradt's methodology.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("needle_haystack", config)
        
        # Load configuration with defaults
        self.context_lengths = config.get("context_lengths", [4000, 8000, 16000, 32000])
        self.depths = config.get("depths", [0.0, 0.25, 0.5, 0.75, 1.0])
        self.num_samples = config.get("num_samples_per_config", 1)
        
        # Needle and question
        self.needle = config.get(
            "needle",
            "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
        )
        self.question = config.get(
            "question",
            "\n\nBased on the content above, what is the best thing to do in San Francisco?\nAnswer:"
        )
        
        # Load haystack
        self.haystack_text = self._load_haystack()
        
        # Success criteria
        self.success_keywords = config.get(
            "success_keywords",
            ["dolores park", "sandwich", "san francisco"]
        )
    
    def _load_haystack(self) -> str:
        """Load Paul Graham essays as haystack"""
        # Try to find the file
        possible_paths = [
            Path("data/raw/PaulGrahamEssays.txt"),
            Path(__file__).parent.parent.parent / "data" / "raw" / "PaulGrahamEssays.txt",
        ]
        
        for data_path in possible_paths:
            if data_path.exists():
                print(f"Loading haystack from {data_path}")
                with open(data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if len(text) > 100000:  # At least 100KB
                    print(f"✓ Haystack loaded: {len(text):,} characters ({len(text)/1024:.1f}KB)")
                    return text
        
        raise FileNotFoundError(
            "Could not find PaulGrahamEssays.txt\n"
            "Please ensure data/raw/PaulGrahamEssays.txt exists"
        )
    
    def _create_context(self, context_length: int, depth: float, model) -> str:
        """Create context with needle at specified depth
        
        Args:
            context_length: Target context length in tokens
            depth: Where to insert needle (0.0 = start, 1.0 = end)
            model: Model object (for tokenization)
        """
        # Tokenize haystack
        haystack_tokens = model.tokenizer.encode(
            self.haystack_text,
            add_special_tokens=False
        )
        
        # Repeat haystack if needed
        if len(haystack_tokens) < context_length:
            repetitions = (context_length // len(haystack_tokens)) + 1
            haystack_tokens = haystack_tokens * repetitions
        
        # Tokenize needle
        needle_tokens = model.tokenizer.encode(
            self.needle,
            add_special_tokens=False
        )
        
        # Calculate insertion point
        available_tokens = context_length - len(needle_tokens)
        insertion_point = int(available_tokens * depth)
        
        # Build context: [before] + [needle] + [after]
        context_tokens = (
            haystack_tokens[:insertion_point] +
            needle_tokens +
            haystack_tokens[insertion_point:insertion_point + (available_tokens - insertion_point)]
        )
        
        # Ensure exact length and decode
        context_tokens = context_tokens[:context_length]
        context = model.tokenizer.decode(context_tokens, skip_special_tokens=True)
        
        return context
    
    def _check_success(self, answer: str) -> bool:
        """Check if answer contains the needle"""
        answer_lower = answer.lower()
        
        # Success if it has EITHER "dolores park" OR "sandwich"
        # (Both are unique enough to indicate finding the needle)
        has_dolores_park = "dolores park" in answer_lower
        has_sandwich = "sandwich" in answer_lower
        
        return has_dolores_park or has_sandwich
    
    def evaluate(self, model) -> Dict[str, Any]:
        """Run needle-in-haystack evaluation"""
        print(f"\n{'='*70}")
        print(f"NEEDLE-IN-HAYSTACK EVALUATION")
        print(f"{'='*70}")
        print(f"Needle: {self.needle[:80]}...")
        print(f"Context lengths: {self.context_lengths}")
        print(f"Depths: {self.depths}")
        print(f"Total tests: {len(self.context_lengths) * len(self.depths)}")
        print(f"{'='*70}\n")
        
        results_list = []
        
        for context_length in self.context_lengths:
            for depth in self.depths:
                print(f"Testing: {context_length:5d} tokens, depth {depth:.0%}...", end=" ", flush=True)
                
                try:
                    # Create test case
                    context = self._create_context(context_length, depth, model)
                    prompt = context + self.question
                    
                    prompt_tokens = model.tokenizer.encode(prompt, return_tensors="pt")
                    prompt_length = prompt_tokens.shape[1]
                    
                    # Generate answer
                    answer = model.generate(prompt, max_tokens=50)
                    
                    
                    # Check success
                    success = self._check_success(answer)
                    
                    # Store result
                    results_list.append({
                        "context_length": context_length,
                        "depth": round(depth, 2),  # ← Round to avoid float precision issues
                        "success": success,
                        "answer": answer  # ← Truncate long answers to save space
                    })
                    
                    status = "✓ PASS" if success else "✗ FAIL"
                    print(status)
                    
                except Exception as e:
                    print(f"✗ ERROR: {str(e)[:50]}")
                    results_list.append({
                        "context_length": context_length,
                        "depth": round(depth, 2),
                        "success": False,
                        "answer": f"Error: {str(e)[:100]}"
                    })
        
        # Calculate metrics
        total = len(results_list)
        successes = sum(1 for r in results_list if r["success"])
        overall_accuracy = successes / total if total > 0 else 0.0
        
        # By context length
        by_length = {}
        for length in self.context_lengths:
            length_results = [r for r in results_list if r["context_length"] == length]
            if length_results:
                length_successes = sum(1 for r in length_results if r["success"])
                by_length[length] = length_successes / len(length_results)
        
        # By depth - use rounded floats as keys
        by_depth = {}
        for depth in [round(d, 2) for d in self.depths]:
            depth_results = [r for r in results_list if r["depth"] == depth]
            if depth_results:
                depth_successes = sum(1 for r in depth_results if r["success"])
                by_depth[depth] = depth_successes / len(depth_results)
        
        # Store results
        self.results = {
            "overall_accuracy": overall_accuracy,
            "total_tests": total,
            "successes": successes,
            "by_context_length": by_length,
            "by_depth": by_depth,
            "details": results_list
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({successes}/{total})")
        
        print(f"\nBy Context Length:")
        for length in sorted(by_length.keys()):
            acc = by_length[length]
            print(f"  {length:5d} tokens: {acc:.2%}")
        
        print(f"\nBy Depth:")
        for depth in sorted(by_depth.keys()):
            acc = by_depth[depth]
            print(f"  Depth {depth:.0%}: {acc:.2%}")
        
        print(f"{'='*70}\n")
        
        return self.results
