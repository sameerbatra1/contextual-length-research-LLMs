# src/evaluators/needle_haystack.py
import json
from pathlib import Path
from typing import Dict, Any, List
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
        self.num_trials = config.get("num_trials", 1)
        
        # Needle and question
        self.needle = config.get(
            "needle",
            "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
        )
        self.question = config.get(
            "question",
            "\n\nBased on the content above, what is the best thing to do in San Francisco?\nAnswer:"
        )
        
        # Pile data configuration
        self.use_pile_data = config.get("use_pile_data", False)
        self.pile_data_file = config.get("pile_data_file", "test_4k_context.json")
        self.pile_data_dir = config.get("pile_data_dir", "data/pile_test_stratified")
        
        # Load haystack
        if self.use_pile_data:
            self.haystack_documents = self._load_pile_documents()
            self.haystack_text = None  # Will be generated per test
        else:
            self.haystack_text = self._load_paul_graham_essays()
            self.haystack_documents = None
        
        # Success criteria
        self.success_keywords = config.get(
            "success_keywords",
            ["dolores park", "sandwich", "san francisco"]
        )
    
    def _load_pile_documents(self) -> List[Dict]:
        """Load documents from Pile test data JSON file"""
        data_path = Path(self.pile_data_dir) / self.pile_data_file
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find Pile data file: {data_path}\n"
                f"Available files should be: test_4k_context.json or test_8k_context.json\n"
                "Please run: python data/data/getting_data.py"
            )
        
        print(f"Loading Pile documents from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"✓ Loaded {len(documents)} documents from Pile dataset")
        if documents:
            print(f"  Document length range: {documents[0]['length_bin']}")
        
        return documents
    
    def _load_paul_graham_essays(self) -> str:
        """Load Paul Graham essays as haystack (fallback)"""
        pg_path = Path(__file__).parent.parent.parent / "data" / "raw" / "PaulGrahamEssays.txt"
        
        if not pg_path.exists():
            raise FileNotFoundError(
                "Could not find PaulGrahamEssays.txt\n"
                "Please ensure data/raw/PaulGrahamEssays.txt exists"
            )
        
        print(f"Loading haystack from {pg_path}")
        with open(pg_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) > 100000:  # At least 100KB
            print(f"✓ Haystack loaded: {len(text):,} characters ({len(text)/1024:.1f}KB)")
            return text
        else:
            raise ValueError("PaulGrahamEssays.txt is too small")
    
    def _get_haystack_text(self, target_tokens: int, model) -> str:
        """Get haystack text from either Pile documents or Paul Graham essays
        
        Args:
            target_tokens: Target number of tokens needed
            model: Model object (for tokenization)
            
        Returns:
            Combined haystack text
        """
        if self.use_pile_data:
            # Select random documents and combine them
            random.shuffle(self.haystack_documents)
            selected_texts = []
            total_tokens = 0
            
            for doc in self.haystack_documents:
                text = doc['text']
                tokens = model.tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
                selected_texts.append(text)
                
                # Stop when we have enough content (120% buffer)
                if total_tokens >= target_tokens * 1.2:
                    break
            
            return "\n\n".join(selected_texts)
        else:
            # Use Paul Graham essays
            return self.haystack_text
    
    def _create_context(self, context_length: int, depth: float, model) -> str:
        """Create context with needle at specified depth
        
        Args:
            context_length: Target context length in tokens
            depth: Where to insert needle (0.0 = start, 1.0 = end)
            model: Model object (for tokenization)
        """
        # Get haystack text
        haystack_text = self._get_haystack_text(context_length, model)
        
        # Tokenize haystack
        haystack_tokens = model.tokenizer.encode(
            haystack_text,
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
        """Run needle-in-haystack evaluation with multiple trials"""
        print(f"\n{'='*70}")
        print(f"NEEDLE-IN-HAYSTACK EVALUATION")
        print(f"{'='*70}")
        print(f"Needle: {self.needle[:80]}...")
        print(f"Context lengths: {self.context_lengths}")
        print(f"Depths: {self.depths}")
        print(f"Trials per test: {self.num_trials}")
        if self.num_trials > 1:
            print(f"Pass criterion: ≥{(self.num_trials + 1) // 2}/{self.num_trials} trials")
        print(f"Total tests: {len(self.context_lengths) * len(self.depths)}")
        print(f"Total trials: {len(self.context_lengths) * len(self.depths) * self.num_trials}")
        print(f"{'='*70}\n")
        
        results_list = []
        
        for context_length in self.context_lengths:
            for depth in self.depths:
                print(f"Testing: {context_length:5d} tokens, depth {depth:.0%}...", end=" ", flush=True)
                
                trial_results = []
                trial_answers = []
                
                # Run multiple trials for this configuration
                for trial in range(self.num_trials):
                    try:
                        # Create test case (regenerated each trial for randomness with Pile data)
                        context = self._create_context(context_length, depth, model)
                        prompt = context + self.question
                        
                        prompt_tokens = model.tokenizer.encode(prompt, return_tensors="pt")
                        prompt_length = prompt_tokens.shape[1]
                        
                        # Generate answer
                        answer = model.generate(prompt, max_tokens=50)
                        
                        # Check success
                        success = self._check_success(answer)
                        
                        trial_results.append(success)
                        trial_answers.append(answer)
                        
                    except Exception as e:
                        trial_results.append(False)
                        trial_answers.append(f"Error: {str(e)[:100]}")
                
                # Determine overall success using majority voting (≥50% pass)
                passes = sum(trial_results)
                required_passes = (self.num_trials + 1) // 2  # Majority: ceil(n/2)
                overall_success = passes >= required_passes
                
                # Store result with trial details
                result = {
                    "context_length": context_length,
                    "depth": round(depth, 2),
                    "success": overall_success,
                    "trials": {
                        "total": self.num_trials,
                        "passed": passes,
                        "failed": self.num_trials - passes,
                        "pass_rate": passes / self.num_trials if self.num_trials > 0 else 0.0,
                        "results": trial_results,
                        "answers": trial_answers
                    }
                }
                
                # For backward compatibility, include first answer in top level
                result["answer"] = trial_answers[0] if trial_answers else ""
                
                results_list.append(result)
                
                # Print status with trial details
                if self.num_trials > 1:
                    status = f"{'✓ PASS' if overall_success else '✗ FAIL'} ({passes}/{self.num_trials})"
                else:
                    status = "✓ PASS" if overall_success else "✗ FAIL"
                print(status)
        
        # Calculate metrics
        total = len(results_list)
        successes = sum(1 for r in results_list if r["success"])
        overall_accuracy = successes / total if total > 0 else 0.0
        
        # Calculate trial statistics
        if self.num_trials > 1:
            total_trials = sum(r["trials"]["total"] for r in results_list)
            total_trial_passes = sum(r["trials"]["passed"] for r in results_list)
            overall_trial_pass_rate = total_trial_passes / total_trials if total_trials > 0 else 0.0
        else:
            total_trials = total
            total_trial_passes = successes
            overall_trial_pass_rate = overall_accuracy
        
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
            "trials_info": {
                "num_trials_per_test": self.num_trials,
                "total_trials": total_trials,
                "total_trial_passes": total_trial_passes,
                "overall_trial_pass_rate": overall_trial_pass_rate,
                "pass_criterion": f">={(self.num_trials + 1) // 2}/{self.num_trials}"
            },
            "by_context_length": by_length,
            "by_depth": by_depth,
            "details": results_list
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({successes}/{total})")
        
        if self.num_trials > 1:
            print(f"\nTrial Statistics:")
            print(f"  Total trials run: {total_trials}")
            print(f"  Individual trial pass rate: {overall_trial_pass_rate:.2%} ({total_trial_passes}/{total_trials})")
            print(f"  Pass criterion: ≥{(self.num_trials + 1) // 2}/{self.num_trials} trials")
        
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
