import json
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Length bins for Quality dataset (in tokens)
LENGTH_BINS = {
    "2k": (1000, 3000),
    "4k": (3000, 6000),
    "8k": (6000, 12000),
}

SAMPLES_PER_BIN = 100


def bin_by_length(token_count: int) -> str:
    """Assign a token count to a length bin"""
    for bin_name, (min_len, max_len) in LENGTH_BINS.items():
        if min_len <= token_count < max_len:
            return bin_name
    return None


def download_and_process_quality():
    """
    Download and preprocess the QuALITY dataset for long-context evaluation.
    
    QuALITY (Question Answering with Long Input Texts, Yes!) is a multiple-choice
    reading comprehension dataset with documents averaging ~5000 tokens.
    """
    output_dir = Path(__file__).parent.parent / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer for length measurement
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1000000
    
    print("Loading QuALITY dataset from HuggingFace...")
    
    try:
        # Load QuALITY dataset
        dataset = load_dataset("emozilla/quality", trust_remote_code=True)
        # Use validation set (test set doesn't have labels)
        test_set = dataset['validation']
        print(f"Loaded {len(test_set)} samples from validation set")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative dataset name...")
        dataset = load_dataset("quality", trust_remote_code=True)
        test_set = dataset['validation']
        print(f"Loaded {len(test_set)} samples from validation set")
    
    binned_data = {"2k": [], "4k": [], "8k": []}
    skipped = 0
    processed = 0
    
    print("\nProcessing samples...")
    
    for idx, example in enumerate(tqdm(test_set, desc="Processing")):
        try:
            # Extract article (document)
            article = example.get('article', '')
            
            # Extract question
            question = example.get('question', '')
            
            # Extract options (multiple choice)
            options = example.get('options', [])
            
            # Extract correct answer (answer is the index of correct option)
            answer_idx = example.get('answer', -1)
            
            # Validate data
            if not article or not question or not options or answer_idx < 0:
                skipped += 1
                continue
            
            if answer_idx >= len(options):
                skipped += 1
                continue
            
            # Count tokens
            token_count = len(tokenizer.encode(article, add_special_tokens=False, truncation=False))
            bin_name = bin_by_length(token_count)
            
            # Add to appropriate bin
            if bin_name is not None and len(binned_data[bin_name]) < SAMPLES_PER_BIN:
                binned_data[bin_name].append({
                    "article_id": example.get('article_id', f"quality_{idx}"),
                    "article_text": article,
                    "article_tokens": token_count,
                    "question": question,
                    "options": options,
                    "correct_answer_idx": answer_idx,
                    "correct_answer": options[answer_idx],
                    "source": "quality_validation"
                })
                processed += 1
            
            # Early exit if all bins are full
            if all(len(binned_data[b]) >= SAMPLES_PER_BIN for b in ["2k", "4k", "8k"]):
                print(f"\nAll bins filled! Stopping early.")
                break
                
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\nProcessed: {processed} samples")
    print(f"Skipped: {skipped} samples")
    
    # Sample from bins if needed
    sampled_data = {}
    for bin_name in ["2k", "4k", "8k"]:
        available = len(binned_data[bin_name])
        if available >= SAMPLES_PER_BIN:
            sampled_data[bin_name] = random.sample(binned_data[bin_name], SAMPLES_PER_BIN)
        else:
            sampled_data[bin_name] = binned_data[bin_name]
    
    # Save data files
    print("\nSaving data files...")
    for bin_name in ["2k", "4k", "8k"]:
        output_file = output_dir / f"quality_{bin_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data[bin_name], f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(sampled_data[bin_name])} samples to {output_file.name}")
    
    # Save metadata
    metadata = {
        "dataset": "QuALITY",
        "random_seed": RANDOM_SEED,
        "samples_per_bin": SAMPLES_PER_BIN,
        "bins": {
            bin_name: {
                "token_range": f"{LENGTH_BINS[bin_name][0]}-{LENGTH_BINS[bin_name][1]}",
                "num_samples": len(sampled_data[bin_name]),
            }
            for bin_name in ["2k", "4k", "8k"]
        },
        "total_samples": sum(len(sampled_data[b]) for b in ["2k", "4k", "8k"]),
        "format": "multiple_choice"
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create README
    readme_content = f"""# QuALITY Test Data

## Overview
This directory contains preprocessed QuALITY test data for long-context evaluation.

## Dataset Info
- **Source**: QuALITY (Question Answering with Long Input Texts, Yes!)
- **Split**: Validation set
- **Tokenizer**: GPT-2 (for length measurement)
- **Random Seed**: {RANDOM_SEED}
- **Format**: Multiple-choice reading comprehension

## Files

### Data Files
- `quality_2k.json`: {len(sampled_data['2k'])} articles (1K-3K tokens)
- `quality_4k.json`: {len(sampled_data['4k'])} articles (3K-6K tokens)
- `quality_8k.json`: {len(sampled_data['8k'])} articles (6K-12K tokens)

### Metadata
- `metadata.json`: Dataset statistics and configuration

## Data Format

Each JSON file contains an array of objects:
```json
{{
  "article_id": "unique_id",
  "article_text": "Full article text...",
  "article_tokens": 4500,
  "question": "Reading comprehension question?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer_idx": 2,
  "correct_answer": "Option C",
  "source": "quality_validation"
}}
```

## Usage

```python
import json

# Load test data
with open('data/quality/quality_4k.json', 'r') as f:
    test_data = json.load(f)

# Use in evaluation
for example in test_data:
    prompt = example['article_text'] + "\\n\\n" + example['question']
    # Generate answer or choose from options
    # Compare with example['correct_answer']
```

## Evaluation Metrics

This dataset is designed for:
- **Accuracy**: Exact match with correct answer
- **Multiple-Choice Selection**: Model selects from 4 options

## Notes

- Fixed test set ensures reproducibility across all models
- All models evaluated on identical data
- Documents sorted by length for systematic evaluation
- Multiple-choice format allows clear accuracy measurement
- Ideal for testing length extrapolation (2K → 4K → 8K)

---
Generated by: `data/data/getting_quality.py`
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\n{'='*70}")
    print("Done! Saved:")
    print(f"{'='*70}")
    for bin_name in ["2k", "4k", "8k"]:
        count = len(sampled_data[bin_name])
        token_range = LENGTH_BINS[bin_name]
        print(f"  {bin_name}: {count:3d} samples ({token_range[0]:,}-{token_range[1]:,} tokens)")
    print(f"  Total: {sum(len(sampled_data[b]) for b in ['2k', '4k', '8k'])} samples")
    print(f"{'='*70}")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    download_and_process_quality()

