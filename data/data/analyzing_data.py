import json
import statistics
from collections import Counter

jsonl_file = "pile_subset.jsonl"

print("Analyzing pile_subset.jsonl...\n")

# Stats to collect
text_lengths = []
sources = Counter()
total_lines = 0
sample_texts = []

with open(jsonl_file, 'r') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(f"Processed {i} examples...")
        
        try:
            example = json.loads(line)
            total_lines += 1
            
            # Extract text and source
            text = example.get('text', '')
            source = example.get('source', 'unknown')
            
            # Track stats
            text_lengths.append(len(text.split()))  # Word count
            sources[source] += 1
            
            # Keep first 3 samples
            if len(sample_texts) < 3:
                sample_texts.append({
                    'source': source,
                    'length': len(text),
                    'preview': text[:200]  # First 200 chars
                })
        except json.JSONDecodeError:
            continue

print(f"\n{'='*60}")
print(f"DATASET STATISTICS")
print(f"{'='*60}")

print(f"\nTotal examples: {total_lines:,}")
print(f"Total tokens (word count): {sum(text_lengths):,}")
print(f"Average tokens per example: {statistics.mean(text_lengths):.1f}")
print(f"Median tokens per example: {statistics.median(text_lengths):.1f}")
print(f"Min tokens: {min(text_lengths)}")
print(f"Max tokens: {max(text_lengths)}")
print(f"Stdev: {statistics.stdev(text_lengths):.1f}")

# Distribution
print(f"\n{'='*60}")
print(f"TEXT LENGTH DISTRIBUTION (by word count)")
print(f"{'='*60}")
length_ranges = [
    (0, 100, "< 100"),
    (100, 500, "100-500"),
    (500, 1000, "500-1k"),
    (1000, 5000, "1k-5k"),
    (5000, 10000, "5k-10k"),
    (10000, float('inf'), "> 10k")
]

for min_len, max_len, label in length_ranges:
    count = sum(1 for l in text_lengths if min_len <= l < max_len)
    pct = (count / total_lines) * 100
    print(f"{label:12s}: {count:,} examples ({pct:5.1f}%)")

# Source distribution
print(f"\n{'='*60}")
print(f"DATA SOURCES")
print(f"{'='*60}")
for source, count in sources.most_common(10):
    pct = (count / total_lines) * 100
    print(f"{source:30s}: {count:,} examples ({pct:5.1f}%)")

# Sample texts
print(f"\n{'='*60}")
print(f"SAMPLE TEXTS")
print(f"{'='*60}")
for i, sample in enumerate(sample_texts, 1):
    print(f"\nSample {i} (from {sample['source']}):")
    print(f"Length: {sample['length']} chars")
    print(f"Preview: {sample['preview']}...")
    print()

# For YaRN: Check if we have long enough sequences
print(f"{'='*60}")
print(f"SUITABILITY FOR YARN EXPERIMENTS")
print(f"{'='*60}")

min_for_yarn = 512  # Minimum useful length for fine-tuning
suitable = sum(1 for l in text_lengths if l >= min_for_yarn)
pct_suitable = (suitable / total_lines) * 100

print(f"Examples >= {min_for_yarn} tokens: {suitable:,} ({pct_suitable:.1f}%)")
print(f"\n✓ Dataset is {'SUITABLE' if pct_suitable > 80 else 'MARGINAL'} for YaRN fine-tuning")
print(f"  (YaRN paper needs long sequences to test context extension)")
