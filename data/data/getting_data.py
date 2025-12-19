from datasets import load_dataset
import json
import os
import random
import hashlib
from datetime import datetime

def download_stratified_pile_data(
    total_docs=6000,
    length_bins=[(1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000)],
    save_dir="./data/pile_train_stratified",
    seed=42
):
    """
    Download using Hugging Face's mirror instead of the-eye.eu
    """
    
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("DOWNLOADING STRATIFIED PILE TRAINING DATA")
    print("="*70)
    
    docs_per_bin = total_docs // len(length_bins)
    print(f"\nConfiguration:")
    print(f"  Total documents target: {total_docs}")
    print(f"  Documents per bin: {docs_per_bin}")
    print(f"  Random seed: {seed}\n")
    
    # Try loading from Hugging Face cache/mirror
    print("Loading The Pile dataset from Hugging Face...")
    try:
        # Use non-streaming first to download to cache
        pile_train = load_dataset(
            "monology/pile-uncopyrighted",  # Alternative hosted version
            split="train",
            streaming=True
        )
        print("✓ Using pile-uncopyrighted mirror")
    except Exception as e:
        print(f"Mirror 1 failed: {e}")
        print("Trying alternative method...")
        
        # Fallback: Try with trust_remote_code
        try:
            pile_train = load_dataset(
                "EleutherAI/pile",
                split="train",
                streaming=True,
                trust_remote_code=True,
                download_mode="force_redownload"
            )
            print("✓ Using EleutherAI/pile with cache")
        except Exception as e2:
            print(f"Both methods failed. Error: {e2}")
            print("\nPlease see alternative solutions below.")
            raise
    
    pile_iter = iter(pile_train.shuffle(seed=seed, buffer_size=10000))
    
    # Rest of your code remains the same
    bins = {f"{min_len}-{max_len}": [] for min_len, max_len in length_bins}
    bin_targets = {f"{min_len}-{max_len}": docs_per_bin for min_len, max_len in length_bins}
    
    print("Collecting documents...\n")
    
    doc_id = 0
    processed = 0
    
    for doc in pile_iter:
        processed += 1
        
        if all(len(bins[k]) >= bin_targets[k] for k in bins.keys()):
            print("\n✓ All bins filled!")
            break
        
        text_length = len(doc['text'].split())
        
        for min_len, max_len in length_bins:
            bin_key = f"{min_len}-{max_len}"
            
            if min_len <= text_length < max_len and len(bins[bin_key]) < docs_per_bin:
                bins[bin_key].append({
                    'id': doc_id,
                    'text': doc['text'],
                    'meta': doc.get('meta', {}),
                    'length_words': text_length,
                    'length_bin': bin_key,
                    'source_index': processed
                })
                doc_id += 1
                
                if doc_id % 250 == 0:
                    status = ", ".join([f"{k}: {len(v)}/{docs_per_bin}" 
                                      for k, v in bins.items()])
                    print(f"  [{doc_id}/{total_docs}] {status}")
                
                break
    
    # Combine and shuffle
    all_docs = []
    for bin_docs in bins.values():
        all_docs.extend(bin_docs)
    
    random.seed(seed)
    random.shuffle(all_docs)
    
    for i, doc in enumerate(all_docs):
        doc['id'] = i
    
    # Save everything (same as before)
    print(f"\n✓ Collected {len(all_docs)} documents")
    
    output_path = f"{save_dir}/train_documents.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    
    with open(output_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Statistics
    bin_stats = {}
    for bin_key in bins.keys():
        bin_docs = [d for d in all_docs if d['length_bin'] == bin_key]
        if bin_docs:
            lengths = [d['length_words'] for d in bin_docs]
            bin_stats[bin_key] = {
                'count': len(bin_docs),
                'mean_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths)
            }
    
    metadata = {
        'dataset_info': {
            'source': 'monology/pile-uncopyrighted',
            'split': 'train',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'configuration': {
            'total_documents': len(all_docs),
            'seed': seed,
            'length_bins': [f"{min_len}-{max_len}" for min_len, max_len in length_bins],
            'docs_per_bin': docs_per_bin
        },
        'statistics': {
            'per_bin': bin_stats
        },
        'file_info': {
            'size_mb': file_size_mb,
            'sha256_hash': file_hash
        }
    }
    
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved to {save_dir}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Hash: {file_hash[:32]}...")
    
    return all_docs, metadata


def load_saved_training_data(data_dir="./data/pile_train_stratified"):
    """
    Load previously saved training data
    
    Args:
        data_dir: Directory containing saved data
    
    Returns:
        documents: List of documents
        metadata: Dataset metadata
    """
    print(f"Loading training data from {data_dir}...")
    
    # Load documents
    with open(f"{data_dir}/train_documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded {len(documents)} documents")
    print(f"  SHA-256: {metadata['file_info']['sha256_hash'][:32]}...")
    
    return documents, metadata


def verify_dataset_integrity(data_dir="./data/pile_train_stratified"):
    """
    Verify dataset integrity by checking hash
    
    Args:
        data_dir: Directory containing saved data
    
    Returns:
        bool: True if integrity check passes
    """
    print("Verifying dataset integrity...")
    
    # Load saved hash
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    saved_hash = metadata['file_info']['sha256_hash']
    
    # Compute current hash
    with open(f"{data_dir}/train_documents.json", 'rb') as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
    
    if saved_hash == current_hash:
        print(f"✓ Integrity check PASSED")
        print(f"  Hash: {current_hash[:32]}...")
        return True
    else:
        print(f"✗ Integrity check FAILED")
        print(f"  Expected: {saved_hash[:32]}...")
        print(f"  Got:      {current_hash[:32]}...")
        return False

def download_stratified_test_data(
    total_docs=1000,  # Smaller for testing
    length_bins=[(1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000)],
    save_dir="./data/pile_test_stratified",
    seed=123  # Different seed than training to avoid overlap
):
    """
    Download test/validation dataset with same stratification as training
    
    Args:
        total_docs: Total test documents (1000 recommended for 6000 train)
        length_bins: Same bins as training for consistency
        save_dir: Where to save test data
        seed: Different seed from training (123 vs 42)
    """
    
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("DOWNLOADING STRATIFIED TEST/VALIDATION DATA")
    print("="*70)
    
    docs_per_bin = total_docs // len(length_bins)
    print(f"\nConfiguration:")
    print(f"  Total documents target: {total_docs}")
    print(f"  Documents per bin: {docs_per_bin}")
    print(f"  Random seed: {seed} (different from training)")
    print(f"  Purpose: Testing/Validation\n")
    
    # Load dataset
    print("Loading dataset from Hugging Face...")
    try:
        pile_val = load_dataset(
            "monology/pile-uncopyrighted",
            split="validation",  # Use validation split for testing
            streaming=True
        )
        print("✓ Using pile-uncopyrighted validation split")
    except Exception as e:
        print(f"Pile validation failed: {e}")
        print("Trying test split...")
        try:
            pile_val = load_dataset(
                "monology/pile-uncopyrighted",
                split="test",
                streaming=True
            )
            print("✓ Using pile-uncopyrighted test split")
        except Exception as e2:
            print(f"Both splits failed. Trying C4...")
            # Fallback to C4
            pile_val = load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True
            )
            print("✓ Using C4 validation split as fallback")
    
    pile_iter = iter(pile_val.shuffle(seed=seed, buffer_size=10000))
    
    # Initialize bins
    bins = {f"{min_len}-{max_len}": [] for min_len, max_len in length_bins}
    bin_targets = {f"{min_len}-{max_len}": docs_per_bin for min_len, max_len in length_bins}
    
    print("Collecting test documents...\n")
    
    doc_id = 0
    processed = 0
    
    for doc in pile_iter:
        processed += 1
        
        if all(len(bins[k]) >= bin_targets[k] for k in bins.keys()):
            print("\n✓ All bins filled!")
            break
        
        text_length = len(doc['text'].split())
        
        for min_len, max_len in length_bins:
            bin_key = f"{min_len}-{max_len}"
            
            if min_len <= text_length < max_len and len(bins[bin_key]) < docs_per_bin:
                bins[bin_key].append({
                    'id': doc_id,
                    'text': doc['text'],
                    'meta': doc.get('meta', {}),
                    'length_words': text_length,
                    'length_bin': bin_key,
                    'source_index': processed
                })
                doc_id += 1
                
                if doc_id % 100 == 0:
                    status = ", ".join([f"{k}: {len(v)}/{docs_per_bin}" 
                                      for k, v in bins.items()])
                    print(f"  [{doc_id}/{total_docs}] {status}")
                
                break
    
    # Combine and shuffle
    all_docs = []
    for bin_docs in bins.values():
        all_docs.extend(bin_docs)
    
    random.seed(seed)
    random.shuffle(all_docs)
    
    for i, doc in enumerate(all_docs):
        doc['id'] = i
    
    print(f"\n✓ Collected {len(all_docs)} test documents")
    
    # Save test data
    output_path = f"{save_dir}/test_documents.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    
    with open(output_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Statistics
    bin_stats = {}
    for bin_key in bins.keys():
        bin_docs = [d for d in all_docs if d['length_bin'] == bin_key]
        if bin_docs:
            lengths = [d['length_words'] for d in bin_docs]
            bin_stats[bin_key] = {
                'count': len(bin_docs),
                'mean_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths)
            }
    
    # Metadata
    metadata = {
        'dataset_info': {
            'source': 'monology/pile-uncopyrighted',
            'split': 'validation',
            'purpose': 'test_evaluation',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'configuration': {
            'total_documents': len(all_docs),
            'seed': seed,
            'length_bins': [f"{min_len}-{max_len}" for min_len, max_len in length_bins],
            'docs_per_bin': docs_per_bin,
            'note': 'Different seed from training to prevent data leakage'
        },
        'statistics': {
            'per_bin': bin_stats
        },
        'file_info': {
            'size_mb': file_size_mb,
            'sha256_hash': file_hash
        },
        'evaluation_config': {
            'target_context_lengths': [4096, 8192],
            'compatible_with_training_seed': 42
        }
    }
    
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create context-specific subsets for 4K and 8K testing
    create_context_specific_subsets(all_docs, save_dir, length_bins)
    
    print(f"\n✓ Saved to {save_dir}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Hash: {file_hash[:32]}...")
    
    # Print bin distribution
    print(f"\nBin distribution:")
    for bin_key, stats in bin_stats.items():
        print(f"  {bin_key:>15}: {stats['count']:>4} docs "
              f"(avg: {stats['mean_length']:>6.1f} words)")
    
    return all_docs, metadata


def create_context_specific_subsets(all_docs, save_dir, length_bins):
    """
    Create subsets optimized for 4K and 8K context testing
    
    Args:
        all_docs: All test documents
        save_dir: Directory to save subsets
        length_bins: Length bin definitions
    """
    
    print("\nCreating context-specific test subsets...")
    
    # 4K context subset (documents in 1000-4000 word range)
    test_4k = [doc for doc in all_docs 
               if 1000 <= doc['length_words'] < 4000]
    
    with open(f"{save_dir}/test_4k_context.json", 'w') as f:
        json.dump(test_4k, f, indent=2)
    
    print(f"  ✓ 4K context subset: {len(test_4k)} documents")
    
    # 8K context subset (documents in 4000-8000 word range)
    test_8k = [doc for doc in all_docs 
               if 4000 <= doc['length_words'] <= 8000]
    
    with open(f"{save_dir}/test_8k_context.json", 'w') as f:
        json.dump(test_8k, f, indent=2)
    
    print(f"  ✓ 8K context subset: {len(test_8k)} documents")
    
    # Save subset metadata
    subset_meta = {
        'test_4k': {
            'count': len(test_4k),
            'word_range': '1000-4000',
            'target_context': 4096,
            'purpose': 'Test models fine-tuned to 4K context'
        },
        'test_8k': {
            'count': len(test_8k),
            'word_range': '4000-8000',
            'target_context': 8192,
            'purpose': 'Test models fine-tuned to 8K context'
        }
    }
    
    with open(f"{save_dir}/subsets_metadata.json", 'w') as f:
        json.dump(subset_meta, f, indent=2)


def load_test_data(data_dir="./data/pile_test_stratified", context_length=None):
    """
    Load test data - either full set or context-specific subset
    
    Args:
        data_dir: Test data directory
        context_length: None (all), 4096, or 8192 for specific subset
    
    Returns:
        documents: List of test documents
        metadata: Dataset metadata
    """
    
    if context_length is None:
        # Load full test set
        print(f"Loading full test set from {data_dir}...")
        with open(f"{data_dir}/test_documents.json", 'r') as f:
            documents = json.load(f)
        print(f"✓ Loaded {len(documents)} test documents")
        
    elif context_length == 4096:
        # Load 4K-specific subset
        print(f"Loading 4K context test subset...")
        with open(f"{data_dir}/test_4k_context.json", 'r') as f:
            documents = json.load(f)
        print(f"✓ Loaded {len(documents)} documents (4K context)")
        
    elif context_length == 8192:
        # Load 8K-specific subset
        print(f"Loading 8K context test subset...")
        with open(f"{data_dir}/test_8k_context.json", 'r') as f:
            documents = json.load(f)
        print(f"✓ Loaded {len(documents)} documents (8K context)")
        
    else:
        raise ValueError(f"Invalid context_length: {context_length}. Use None, 4096, or 8192")
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return documents, metadata


def verify_no_overlap_with_training(
    train_dir="./data/pile_train_stratified",
    test_dir="./data/pile_test_stratified"
):
    """
    Verify that test and training sets don't overlap
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory
    
    Returns:
        bool: True if no overlap found
    """
    
    print("Checking for data leakage between train/test...")
    
    # Load both datasets
    with open(f"{train_dir}/train_documents.json", 'r') as f:
        train_docs = json.load(f)
    
    with open(f"{test_dir}/test_documents.json", 'r') as f:
        test_docs = json.load(f)
    
    # Create set of training text hashes
    train_hashes = set()
    for doc in train_docs:
        text_hash = hashlib.md5(doc['text'].encode()).hexdigest()
        train_hashes.add(text_hash)
    
    # Check test documents
    overlaps = 0
    for doc in test_docs:
        text_hash = hashlib.md5(doc['text'].encode()).hexdigest()
        if text_hash in train_hashes:
            overlaps += 1
    
    if overlaps == 0:
        print(f"✓ No overlap found between {len(train_docs)} train and {len(test_docs)} test docs")
        return True
    else:
        print(f"✗ WARNING: {overlaps} overlapping documents found!")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Download and save stratified dataset
    print("Starting data collection...\n")
    
    # train_documents, metadata = download_stratified_pile_data(
    #     total_docs=6000,
    #     length_bins=[(1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000)],
    #     save_dir="./data/pile_train_stratified",
    #     seed=42
    # )
    
    # print("\nVerifying saved data...")
    
    # # Verify integrity
    # verify_dataset_integrity("./data/pile_train_stratified")
    
    # print("\n" + "="*70)
    # print("You can now use this data for training:")
    # print("  docs, meta = load_saved_training_data('./data/pile_train_stratified')")
    # print("="*70)
    test_documents, test_metadata = download_stratified_test_data(
    total_docs=1000,  # 1000 test docs for 6000 train docs (standard ratio)
    length_bins=[(1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000)],
    save_dir="./data/pile_test_stratified",
    seed=123  # Different from training seed (42)
    )
    print("\n" + "="*70)
    print("Test data collected successfully!")
    print(f"  Total documents: {len(test_documents)}")
    print(f"  SHA-256 hash: {test_metadata['file_info']['sha256_hash'][:32]}...")
    print("="*70)
    
    # Verify no overlap with training
    if verify_no_overlap_with_training():
        print("\n✓ No overlap found between train/test")
    else:
        print("\n✗ WARNING: Potential data leakage detected!")
    