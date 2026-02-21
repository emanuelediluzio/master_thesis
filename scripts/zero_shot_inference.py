import os
import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def main():
    print("Starting Zero-shot Inference with Gemma-2-9b-it...")
    
    # Configuration
    model_id = "google/gemma-2-9b-it"
    dataset_path = "/work/tesi_ediluzio/data/jsonl/test_dataset_200.jsonl"
    output_path = "/work/tesi_ediluzio/outputs/inference/zero_shot_results.jsonl"
    max_examples = 200 # Run on all 200 test examples
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    dataset = dataset[:max_examples]
    print(f"Processing {len(dataset)} examples...")
    
    results = []
    
    for i, example in enumerate(tqdm(dataset)):
        svg_content = ""
        ground_truth = ""
        
        # Extract SVG from input (assuming format "SVG Content:\n<svg>...")
        if "input" in example and example["input"].startswith("SVG Content:\n"):
            svg_content = example["input"][len("SVG Content:\n"):].strip()
        else:
            svg_content = example.get("input", "")
            
        if "output" in example:
            ground_truth = example["output"].strip()
            
        # Construct Zero-shot Prompt
        # We use a simple instruction prompt for the instruction-tuned model
        prompt = f"User: Describe the following SVG image in detail.\n\n{svg_content}\n\nModel:"
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False, # Greedy decoding for reproducibility
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove prompt)
            # Simple heuristic: take everything after "Model:"
            if "Model:" in generated_text:
                response = generated_text.split("Model:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            response = f"Error: {str(e)}"
            
        result = {
            "id": i,
            "svg": svg_content,
            "ground_truth": ground_truth,
            "generated_caption": response,
            "model": "Zero-shot Gemma-2-9b-it"
        }
        results.append(result)
        
        if i % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
