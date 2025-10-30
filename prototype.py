import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from dotenv import load_dotenv

load_dotenv()
def test_baseline_generation():
    print("Loading Gemma-7B...")
    model_name = "google/gemma-7b"
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("✓ Baseline generation works!")
    
    return model, tokenizer

def apply_linear_pi(model, original_length=8192, target_length=16384):
    """Apply Linear Position Interpolation to model"""
    scaling_factor = target_length / original_length
    
    print(f"\nApplying Linear PI: {original_length} → {target_length}")
    print(f"Scaling factor: {scaling_factor}")
    
    model.config.rope_scaling = {
        "type": "linear",
        "factor": scaling_factor
    }
    
    print("✓ Linear PI applied!")
    return model

def simple_needle_test(model, tokenizer, context_length=2000):
    """Ultra-simple needle in haystack test"""
    needle = "The secret code is 42"
    filler = "This is random text. " * 1000
    
    haystack = filler[:context_length//2] + needle + filler[context_length//2:]
    
    query = "\n\nQuestion: What is the secret code?\nAnswer:"
    full_prompt = haystack + query
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    success = "42" in answer
    
    print(f"\nNeedle Test (context: {context_length} chars)")
    print(f"Needle: {needle}")
    print(f"Answer: {answer[-100:]}")  
    print(f"Success: {success}")
    
    return success

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model, tokenizer = test_baseline_generation()
    
    model = apply_linear_pi(model, original_length=8192, target_length=32768)
    
    simple_needle_test(model, tokenizer, context_length=2000)
    
    print("\n✓ Prototype complete! Ready to build OOP structure.")
    # from transformers import AutoModelForCausalLM, AutoConfig
    # # Load Gemma config first (lightweight)
    # config = AutoConfig.from_pretrained("google/gemma-7b", token=token)

    # print("Model type:", config.model_type)
    # print("Max position embeddings:", config.max_position_embeddings)
    # print("Has rope_scaling:", hasattr(config, 'rope_scaling'))
    # print("Current rope_scaling:", getattr(config, 'rope_scaling', None))
