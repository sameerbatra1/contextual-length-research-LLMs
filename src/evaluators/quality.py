# src/evaluators/quality.py
"""
QuALITY Evaluator for Long-Context Reading Comprehension

This evaluator tests models on multiple-choice reading comprehension questions from the QuALITY dataset.
QuALITY (Question Answering with Long Input Texts, Yes!) provides high-quality questions
about long articles (2K-8K tokens).

Metrics:
- Accuracy: Percentage of correctly answered questions
- By context length: Performance breakdown by document length
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import random
import re

from .base_evaluator import BaseEvaluator


class QualityEvaluator(BaseEvaluator):
    """
    QuALITY evaluation for long-form reading comprehension.
    
    Tests model's ability to:
    - Read and comprehend long articles (2K-8K tokens)
    - Answer multiple-choice questions about content
    - Select correct answer from 4 options
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("quality", config)
        
        # Load configuration
        self.context_lengths = config.get("context_lengths", [2048, 4096, 8192])
        self.num_samples = config.get("num_samples", 100)  # Samples per length
        self.num_trials = config.get("num_trials", 1)
        
        # Data configuration
        self.data_dir = config.get("data_dir", "data/quality")
        self.data_files = config.get("data_files", {
            2048: "quality_2k.json",
            4096: "quality_4k.json",
            8192: "quality_8k.json"
        })
        
        # Load test data
        self.test_data = self._load_test_data()
        
        print(f"✓ QuALITY Evaluator initialized")
        print(f"  Context lengths: {self.context_lengths}")
        print(f"  Samples per length: {self.num_samples}")
        print(f"  Total test cases: {sum(len(self.test_data.get(length, [])) for length in self.context_lengths)}")
    
    def _load_test_data(self) -> Dict[int, List[Dict]]:
        """Load preprocessed QuALITY test data"""
        test_data = {}
        data_dir = Path(self.data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"QuALITY data directory not found: {data_dir}\n"
                f"Please run: python data/data/getting_quality.py"
            )
        
        for context_length in self.context_lengths:
            if context_length in self.data_files:
                data_file = data_dir / self.data_files[context_length]
                
                if not data_file.exists():
                    print(f"⚠ Warning: Data file not found: {data_file}")
                    test_data[context_length] = []
                    continue
                
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Limit to num_samples if specified
                if self.num_samples and len(data) > self.num_samples:
                    data = data[:self.num_samples]
                
                test_data[context_length] = data
                print(f"  Loaded {len(data)} samples for {context_length} tokens from {data_file.name}")
            else:
                print(f"⚠ Warning: No data file specified for context length {context_length}")
                test_data[context_length] = []
        
        return test_data
    
    def _create_prompt(self, article: str, question: str, options: List[str], 
                       context_length: int, model) -> str:
        """
        Create prompt for the model, truncating article to fit context_length.
        Format: Article + Question + Options
        """
        # Format options
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        question_prompt = f"\n\nQuestion: {question}\n\n{options_text}\n\nAnswer:"
        
        # Calculate token lengths
        question_tokens = model.tokenizer.encode(question_prompt, add_special_tokens=False)
        question_length = len(question_tokens)
        
        # Reserve additional space for generation (50 tokens) and safety margin (50 tokens)
        safety_margin = 100
        
        # Calculate available space for article
        available_for_article = context_length - question_length - safety_margin
        
        if available_for_article <= 0:
            # Question is too long, use minimal article
            available_for_article = 500
        
        # Tokenize and truncate article to fit
        article_tokens = model.tokenizer.encode(article, add_special_tokens=False)
        
        if len(article_tokens) > available_for_article:
            # Truncate article
            article_tokens = article_tokens[:available_for_article]
            article = model.tokenizer.decode(article_tokens, skip_special_tokens=True)
        
        # Create final prompt
        prompt = f"{article}{question_prompt}"
        
        return prompt
    
    def _parse_answer(self, generated: str, options: List[str]) -> int:
        """
        Parse the generated answer to extract the selected option index.
        
        Handles various formats:
        - "A", "B", "C", "D" (letter only)
        - "The answer is B"
        - "B. Option text"
        - Full option text matching
        
        Returns: index of selected option (0-3), or -1 if cannot parse
        """
        if not generated or not generated.strip():
            return -1
        
        generated = generated.strip()
        
        # Method 1: Look for letter (A, B, C, D) in the first part of the answer
        first_line = generated.split('\n')[0].strip()
        
        # Try to find a single letter A-D
        letter_match = re.search(r'\b([A-D])\b', first_line.upper())
        if letter_match:
            letter = letter_match.group(1)
            return ord(letter) - ord('A')
        
        # Method 2: Check if full option text appears in the answer
        for idx, option in enumerate(options):
            option_lower = option.lower().strip()
            generated_lower = generated.lower()
            
            # Check if option text is in the generated answer
            if option_lower in generated_lower:
                return idx
        
        # Method 3: Try to find number (1, 2, 3, 4)
        number_match = re.search(r'\b([1-4])\b', first_line)
        if number_match:
            number = int(number_match.group(1))
            return number - 1
        
        # Could not parse
        return -1
    
    def evaluate(self, model) -> Dict[str, Any]:
        """Run QuALITY evaluation"""
        print(f"\n{'='*70}")
        print(f"QUALITY EVALUATION")
        print(f"{'='*70}")
        print(f"Context lengths: {self.context_lengths}")
        print(f"Samples per length: {self.num_samples}")
        print(f"Trials per sample: {self.num_trials}")
        print(f"Metrics: Accuracy (Multiple Choice)")
        
        total_tests = sum(len(self.test_data.get(length, [])) for length in self.context_lengths)
        total_trials = total_tests * self.num_trials
        
        print(f"Total tests: {total_tests}")
        print(f"Total trials: {total_trials}")
        print(f"{'='*70}\n")
        
        results_list = []
        
        for context_length in self.context_lengths:
            test_samples = self.test_data.get(context_length, [])
            
            if not test_samples:
                print(f"⚠ Skipping {context_length} tokens (no data)")
                continue
            
            print(f"\nTesting {context_length} tokens ({len(test_samples)} samples)...")
            print("-" * 70)
            
            for sample_idx, sample in enumerate(test_samples):
                print(f"  [{sample_idx+1}/{len(test_samples)}] ", end="", flush=True)
                
                trial_results = []
                trial_answers = []
                trial_parsed_indices = []
                
                # Run multiple trials
                for trial in range(self.num_trials):
                    try:
                        # Create prompt (with truncation)
                        prompt = self._create_prompt(
                            sample['article_text'],
                            sample['question'],
                            sample['options'],
                            context_length,
                            model
                        )
                        
                        # Generate answer (shorter generation for multiple choice)
                        answer = model.generate(prompt, max_tokens=50)
                        
                        # Parse answer to get selected option index
                        selected_idx = self._parse_answer(answer, sample['options'])
                        
                        # Check if correct
                        correct = (selected_idx == sample['correct_answer_idx'])
                        
                        trial_answers.append(answer)
                        trial_parsed_indices.append(selected_idx)
                        trial_results.append(correct)
                        
                    except Exception as e:
                        trial_answers.append(f"Error: {str(e)[:100]}")
                        trial_parsed_indices.append(-1)
                        trial_results.append(False)
                
                # Aggregate trial results
                passes = sum(trial_results)
                required_passes = (self.num_trials + 1) // 2
                overall_success = passes >= required_passes
                
                # Store result
                result = {
                    "article_id": sample['article_id'],
                    "context_length": context_length,
                    "article_tokens": sample['article_tokens'],
                    "question": sample['question'][:100] + "..." if len(sample['question']) > 100 else sample['question'],
                    "options": sample['options'],
                    "correct_answer_idx": sample['correct_answer_idx'],
                    "correct_answer": sample['correct_answer'],
                    "success": overall_success,
                    "trials": {
                        "total": self.num_trials,
                        "passed": passes,
                        "parsed_indices": trial_parsed_indices,
                        "raw_answers": trial_answers
                    }
                }
                
                results_list.append(result)
                
                # Print inline status
                if self.num_trials > 1:
                    status = f"{'✓' if overall_success else '✗'} ({passes}/{self.num_trials})"
                else:
                    status = f"{'✓' if overall_success else '✗'}"
                print(status)
        
        # Calculate aggregate metrics
        total = len(results_list)
        correct = sum(1 for r in results_list if r["success"])
        overall_accuracy = correct / total if total > 0 else 0.0
        
        # By context length
        by_length = {}
        for length in self.context_lengths:
            length_results = [r for r in results_list if r["context_length"] == length]
            if length_results:
                by_length[length] = {
                    "num_samples": len(length_results),
                    "correct": sum(1 for r in length_results if r["success"]),
                    "accuracy": sum(1 for r in length_results if r["success"]) / len(length_results),
                }
        
        # Store results
        self.results = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total,
            "correct": correct,
            "by_context_length": by_length,
            "details": results_list
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        
        print(f"\nBy Context Length:")
        for length in sorted(by_length.keys()):
            stats = by_length[length]
            print(f"  {length:5d} tokens:")
            print(f"    Samples:   {stats['num_samples']}")
            print(f"    Correct:   {stats['correct']}")
            print(f"    Accuracy:  {stats['accuracy']:.2%}")
        
        print(f"{'='*70}\n")
        
        return self.results

