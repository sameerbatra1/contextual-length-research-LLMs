# src/evaluators/perplexity.py
"""
Perplexity Evaluator for Long-Context Language Modeling

This evaluator tests models on perplexity across different context lengths using the Pile dataset.
Perplexity measures how well a language model predicts the next token in a sequence.

Metrics:
- Perplexity: exp(average negative log-likelihood) across documents
- By context length: Performance breakdown by document length
- Sliding window: Perplexity across different positions in long documents
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, List
import random
import math

from .base_evaluator import BaseEvaluator


class PerplexityEvaluator(BaseEvaluator):
    """
    Perplexity evaluation for long-context language modeling.
    
    Tests model's ability to:
    - Predict next tokens in long documents
    - Maintain coherence across different context lengths
    - Handle varying document domains from Pile dataset
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("perplexity", config)
        
        # Load configuration
        self.context_lengths = config.get("context_lengths", [2048, 4096, 8192])
        self.num_samples = config.get("num_samples", 50)  # Samples per length
        self.stride = config.get("stride", 512)  # Stride for sliding window
        self.max_tokens_per_doc = config.get("max_tokens_per_doc", None)  # Limit tokens per doc
        self.seed = config.get("seed", 42)  # Random seed for reproducibility
        
        # Evaluation mode: 'truncate' or 'sliding_window'
        # truncate: Cut documents to exact context_length (conservative, tests within-context)
        # sliding_window: Allow longer docs with sliding window (standard, tests full capability)
        self.eval_mode = config.get("eval_mode", "sliding_window")
        
        # Data configuration
        self.data_dir = config.get("data_dir", "data/pile_test_stratified")
        self.data_file = config.get("data_file", "test_documents.json")
        
        # Length bins mapping - use longest docs for all contexts
        # Documents capped at max(context_lengths) to fit model's extended capability
        # Sliding windows evaluate different context lengths on the same long documents
        self.length_bin_mapping = {
            2048: "6000-8000",  # ~8K tokens → multiple windows at 2048 context
            4096: "6000-8000",  # ~8K tokens → fewer windows at 4096 context
            8192: "6000-8000"   # ~8K tokens → 1 window at 8192 context
        }
        
        # Load test data
        self.test_data = self._load_test_data()
        
        print(f"✓ Perplexity Evaluator initialized")
        print(f"  Context lengths: {self.context_lengths}")
        print(f"  Samples per length: {self.num_samples}")
        print(f"  Stride: {self.stride}")
        print(f"  Eval mode: {self.eval_mode}")
        if self.eval_mode == "truncate":
            print(f"    → Documents truncated to exact context length")
        elif self.max_tokens_per_doc:
            print(f"    → Sliding window, max {self.max_tokens_per_doc} tokens")
        else:
            max_len = max(self.context_lengths)
            print(f"    → Sliding window, documents capped at {max_len} tokens (max context)")
            print(f"    → Same long documents evaluated with different window sizes")
        print(f"  Document bin: '6000-8000' words (~8-10K tokens, capped)")
        print(f"  Random seed: {self.seed} (for reproducibility)")
        print(f"  Total test cases: {sum(len(self.test_data.get(length, [])) for length in self.context_lengths)}")
    
    def _load_test_data(self) -> Dict[int, List[Dict]]:
        """Load Pile test documents and organize by context length"""
        data_path = Path(self.data_dir) / self.data_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"Pile test data not found at {data_path}")
        
        print(f"\nLoading Pile test data from {data_path}...")
        
        with open(data_path, 'r') as f:
            all_documents = json.load(f)
        
        # Organize by context length
        organized_data = {}
        
        for idx, context_length in enumerate(self.context_lengths):
            target_bin = self.length_bin_mapping.get(context_length)
            
            if target_bin:
                # Filter documents by length bin
                bin_docs = [
                    doc for doc in all_documents 
                    if doc.get('length_bin') == target_bin
                ]
                
                # Sample randomly with deterministic seed
                # Use different seed for each context length to ensure diversity
                if len(bin_docs) > self.num_samples:
                    random.seed(self.seed + idx)  # Reproducible but different per length
                    bin_docs = random.sample(bin_docs, self.num_samples)
                
                organized_data[context_length] = bin_docs
                print(f"  {context_length} tokens: {len(bin_docs)} documents from bin '{target_bin}' (seed={self.seed + idx})")
            else:
                print(f"  ⚠ No length bin mapping for {context_length}")
                organized_data[context_length] = []
        
        return organized_data
    
    def _calculate_perplexity(self, model, text: str, context_length: int, input_ids=None) -> Dict[str, Any]:
        """
        Calculate perplexity for a given text using sliding window approach.
        
        Args:
            model: Model wrapper with tokenizer and device
            text: Text to evaluate (used if input_ids is None)
            context_length: Context window size
            input_ids: Pre-tokenized input (optional, to avoid re-tokenization)
        
        Returns metrics including overall perplexity and position-based perplexity.
        """
        # Use pre-tokenized input if provided, otherwise tokenize
        if input_ids is None:
            # Tokenize the full text WITHOUT special tokens (BOS/EOS)
            # This is standard for perplexity evaluation to avoid predicting BOS with no context
            encodings = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids.to(model.device)
        else:
            # Ensure it's a tensor on the right device
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids], device=model.device)
            elif input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(model.device)
        
        seq_len = input_ids.size(1)
        
        # If text is shorter than context_length, evaluate it as-is
        if seq_len <= context_length:
            with torch.no_grad():
                outputs = model.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                perplexity = math.exp(loss)
            
            return {
                "perplexity": perplexity,
                "num_tokens": seq_len,
                "num_windows": 1,
                "avg_loss": loss
            }
        
        # Sliding window approach for longer texts
        window_losses = []
        num_windows = 0
        
        # Use stride for sliding window
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + context_length, seq_len)
            trg_len = end_loc - begin_loc
            
            input_ids_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_window.clone()
            
            with torch.no_grad():
                outputs = model.model(input_ids_window, labels=target_ids)
                # outputs.loss is already the mean loss per token in this window
                window_losses.append(outputs.loss)
            
            num_windows += 1
            
            # Stop if we've reached the end
            if end_loc == seq_len:
                break
        
        # Calculate overall perplexity: MEAN of window losses (not sum)
        # This avoids double-counting overlapping tokens
        avg_loss = torch.stack(window_losses).mean().item()
        perplexity = math.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "num_tokens": seq_len,
            "num_windows": num_windows,
            "avg_loss": avg_loss
        }
    
    def evaluate(self, model) -> Dict[str, Any]:
        """Run Perplexity evaluation"""
        print(f"\n{'='*70}")
        print(f"PERPLEXITY EVALUATION")
        print(f"{'='*70}")
        print(f"Context lengths: {self.context_lengths}")
        print(f"Samples per length: {self.num_samples}")
        print(f"Stride: {self.stride}")
        print(f"Metrics: Perplexity (lower is better)")
        
        total_tests = sum(len(self.test_data.get(length, [])) for length in self.context_lengths)
        
        print(f"Total tests: {total_tests}")
        print(f"{'='*70}\n")
        
        results_list = []
        total_tokens_processed = 0
        total_windows_processed = 0
        
        for context_length in self.context_lengths:
            test_samples = self.test_data.get(context_length, [])
            
            if not test_samples:
                print(f"⚠ Skipping {context_length} tokens (no data)")
                continue
            
            print(f"\nTesting {context_length} tokens ({len(test_samples)} samples)...")
            print("-" * 70)
            
            perplexities = []
            
            for sample_idx, sample in enumerate(test_samples):
                print(f"  [{sample_idx+1}/{len(test_samples)}] ", end="", flush=True)
                
                try:
                    text = sample['text']
                    
                    # Tokenize WITHOUT special tokens (standard for perplexity)
                    tokens = model.tokenizer.encode(text, add_special_tokens=False)
                    original_length = len(tokens)
                    was_truncated = False
                    
                    # Determine max length based on eval mode
                    if self.eval_mode == "truncate":
                        # Truncate to exact context length (conservative, within-context only)
                        max_tokens = context_length
                    elif self.max_tokens_per_doc:
                        # Use explicit limit if provided
                        max_tokens = self.max_tokens_per_doc
                    else:
                        # Sliding window mode: Cap at MAXIMUM context length across all tests
                        # This ensures all tests use same long documents with different window sizes
                        # E.g., 8K-token doc evaluated with 2048, 4096, and 8192 windows
                        max_tokens = max(self.context_lengths)
                    
                    if len(tokens) > max_tokens:
                        tokens = tokens[:max_tokens]
                        was_truncated = True
                    
                    # Calculate perplexity using the tokens directly
                    # Pass tokens to avoid re-tokenization issues
                    metrics = self._calculate_perplexity(model, text, context_length, input_ids=tokens)
                    metrics['was_truncated'] = was_truncated
                    metrics['original_length'] = original_length
                    metrics['eval_mode'] = self.eval_mode
                    
                    perplexities.append(metrics['perplexity'])
                    
                    # Track totals
                    total_tokens_processed += metrics['num_tokens']
                    total_windows_processed += metrics['num_windows']
                    
                    # Store detailed result
                    result = {
                        "doc_id": sample.get('id', sample_idx),
                        "context_length": context_length,
                        "length_bin": sample.get('length_bin', 'unknown'),
                        "perplexity": metrics['perplexity'],
                        "num_tokens": metrics['num_tokens'],
                        "num_windows": metrics['num_windows'],
                        "avg_loss": metrics['avg_loss'],
                        "was_truncated": metrics.get('was_truncated', False),
                        "original_length": metrics.get('original_length', metrics['num_tokens'])
                    }
                    
                    results_list.append(result)
                    
                    truncated_marker = " [T]" if metrics.get('was_truncated') else ""
                    print(f"PPL: {metrics['perplexity']:.2f} ({metrics['num_tokens']} tok, {metrics['num_windows']} win{truncated_marker})")
                    
                except Exception as e:
                    print(f"Error: {str(e)[:50]}")
                    result = {
                        "doc_id": sample.get('id', sample_idx),
                        "context_length": context_length,
                        "length_bin": sample.get('length_bin', 'unknown'),
                        "perplexity": float('inf'),
                        "error": str(e)[:200]
                    }
                    results_list.append(result)
            
            # Calculate summary statistics for this context length
            if perplexities:
                avg_ppl = sum(perplexities) / len(perplexities)
                min_ppl = min(perplexities)
                max_ppl = max(perplexities)
                
                # Count truncations and token stats
                length_results = [r for r in results_list if r['context_length'] == context_length]
                num_truncated = sum(1 for r in length_results if r.get('was_truncated', False))
                total_tokens_this_length = sum(r['num_tokens'] for r in length_results)
                total_windows_this_length = sum(r['num_windows'] for r in length_results)
                avg_tokens_per_doc = total_tokens_this_length / len(length_results)
                avg_windows_per_doc = total_windows_this_length / len(length_results)
                
                print(f"\n  Summary for {context_length} tokens:")
                print(f"    Average Perplexity: {avg_ppl:.2f}")
                print(f"    Min Perplexity: {min_ppl:.2f}")
                print(f"    Max Perplexity: {max_ppl:.2f}")
                print(f"    Samples: {len(perplexities)}")
                print(f"    Total tokens processed: {total_tokens_this_length:,}")
                print(f"    Total windows: {total_windows_this_length:,}")
                print(f"    Avg tokens/doc: {avg_tokens_per_doc:.0f}")
                print(f"    Avg windows/doc: {avg_windows_per_doc:.1f}")
                if num_truncated > 0:
                    print(f"    Truncated: {num_truncated}/{len(test_samples)} documents")
        
        # Calculate overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        
        summary_by_length = {}
        for context_length in self.context_lengths:
            length_results = [r for r in results_list if r['context_length'] == context_length]
            valid_ppls = [r['perplexity'] for r in length_results if r['perplexity'] != float('inf')]
            
            if valid_ppls:
                num_truncated = sum(1 for r in length_results if r.get('was_truncated', False))
                total_tokens_this_length = sum(r['num_tokens'] for r in length_results)
                total_windows_this_length = sum(r['num_windows'] for r in length_results)
                
                summary_by_length[context_length] = {
                    "average_perplexity": sum(valid_ppls) / len(valid_ppls),
                    "min_perplexity": min(valid_ppls),
                    "max_perplexity": max(valid_ppls),
                    "num_samples": len(valid_ppls),
                    "num_errors": len(length_results) - len(valid_ppls),
                    "num_truncated": num_truncated,
                    "total_tokens": total_tokens_this_length,
                    "total_windows": total_windows_this_length
                }
                
                print(f"\n{context_length} tokens:")
                print(f"  Average PPL: {summary_by_length[context_length]['average_perplexity']:.2f}")
                print(f"  Min PPL: {summary_by_length[context_length]['min_perplexity']:.2f}")
                print(f"  Max PPL: {summary_by_length[context_length]['max_perplexity']:.2f}")
                print(f"  Samples: {summary_by_length[context_length]['num_samples']}")
                print(f"  Total tokens: {total_tokens_this_length:,}")
                print(f"  Total windows: {total_windows_this_length:,}")
                if num_truncated > 0:
                    print(f"  Truncated: {num_truncated} documents")
        
        print(f"\n{'='*70}")
        print(f"GRAND TOTAL: {total_tokens_processed:,} tokens processed across {total_windows_processed:,} windows")
        print(f"{'='*70}\n")
        
        # Compile final results
        final_results = {
            "experiment_info": {
                "context_lengths": self.context_lengths,
                "num_samples_per_length": self.num_samples,
                "stride": self.stride,
                "eval_mode": self.eval_mode,
                "total_documents": len(results_list),
                "total_tokens_processed": total_tokens_processed,
                "total_windows_processed": total_windows_processed
            },
            "summary": summary_by_length,
            "detailed_results": results_list
        }
        
        self.results = final_results
        return final_results

