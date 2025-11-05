import os
import json
from huggingface_hub import hf_hub_download
import zstandard as zstd
import io

output_file = "./data/pile_subset.jsonl"
os.makedirs("./data/raw", exist_ok=True)

repo_id = "monology/pile-uncopyrighted"
example_count = 0

with open(output_file, 'w') as out_f:
    for file_num in range(5):
        file_path = f"train/{file_num:02d}.jsonl.zst"
        print(f"\n[{file_num + 1}/5] Downloading {file_path}...")
        
        # Download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir="./data/raw"
        )
        
        print(f"[{file_num + 1}/5] Processing...")
        dctx = zstd.ZstdDecompressor()
        
        with open(downloaded_path, 'rb') as f_in:
            reader = dctx.stream_reader(f_in, closefd=False)
            # Wrap in TextIOWrapper to iterate lines
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            for line in text_stream:
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        out_f.write(json.dumps(example) + '\n')
                        example_count += 1
                        
                        if example_count % 50000 == 0:
                            print(f"  Processed {example_count} examples...")
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        pass

print(f"\n✓ Total: {example_count} examples")
