# src/evaluators/needle_haystack.py
from .base_evaluator import BaseEvaluator
from typing import Dict
import random

class NeedleHaystackEvaluator(BaseEvaluator):
    """Needle in a haystack evaluation"""
    
    def __init__(self, config: Dict = None):
        super().__init__("needle_haystack", config)
        
        # Default configuration
        self.depths = config.get("depths", [0.0, 0.5, 1.0]) if config else [0.0, 0.5, 1.0]
        self.context_lengths = config.get("context_lengths", [1000, 2000]) if config else [1000, 2000]
        self.num_samples = config.get("num_samples", 1) if config else 1
        
        # Needle and haystack
        self.needle = "The secret password is 'strawberry42'."
        self.question = "\n\nQuestion: What is the secret password?\nAnswer: The secret password is '"
        self.filler_text = self._generate_filler()
    
    def _generate_filler(self) -> str:
        """Generate filler text for haystack"""
        # Simple approach: repeat some text
        base_text = """The quick brown fox jumps over the lazy dog. 
        This is a test of the emergency broadcast system. 
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. """
        return base_text * 1000  # Make it long enough
    
    def _create_haystack(self, length: int, depth: float) -> str:
        """Create haystack with needle at specified depth"""
        # Calculate insertion point
        insertion_point = int(length * depth)
        
        # Create haystack
        before = self.filler_text[:insertion_point]
        after = self.filler_text[insertion_point:insertion_point + length]
        
        haystack = before + self.needle + after
        return haystack[:length]  # Trim to exact length
    
    def evaluate(self, model) -> Dict[str, float]:
        """Run needle-in-haystack evaluation"""
        print(f"\n{'='*60}")
        print(f"Running Needle-in-Haystack Evaluation")
        print(f"{'='*60}")
        
        results_list = []
        
        for length in self.context_lengths:
            for depth in self.depths:
                print(f"\nTesting: length={length}, depth={depth:.1%}")
                
                # Create test case
                haystack = self._create_haystack(length, depth)
                prompt = haystack + self.question
                
                # Generate answer
                response = model.generate(prompt, max_tokens=20)
                
                # Extract answer (text after the question)
                answer = response.split("Answer:")[-1].strip()
                
                # Check if correct
                success = "strawberry42" in answer.lower()
                
                results_list.append({
                    "length": length,
                    "depth": depth,
                    "success": success,
                    "answer": answer[:50]  # Store first 50 chars
                })
                
                status = "✓" if success else "✗"
                print(f"  {status} Answer: {answer[:50]}")
        
        # Calculate aggregate metrics
        total = len(results_list)
        successes = sum(1 for r in results_list if r["success"])
        accuracy = successes / total if total > 0 else 0.0
        
        self.results = {
            "accuracy": accuracy,
            "total_tests": total,
            "successes": successes,
            "details": results_list
        }
        
        print(f"\n{'='*60}")
        print(f"Overall Accuracy: {accuracy:.2%} ({successes}/{total})")
        print(f"{'='*60}")
        
        return self.results
