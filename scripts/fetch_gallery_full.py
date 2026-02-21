import json
import os
import glob

json_dir = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/json"
target_ids = [17, 26, 12, 11, 22, 18, 25, 15, 16, 24, 27, 9, 14, 21, 13, 19]

files = glob.glob(os.path.join(json_dir, "*.json"))
results = {tid: {} for tid in target_ids}

print(f"Scanning {len(files)} files for Gallery IDs...")

for filepath in files:
    model_name = os.path.basename(filepath).replace("_results.json", "")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "results" in data:
            items = data["results"]
            
        for item in items:
            sid = item.get("sample_id") or item.get("id")
            if sid in target_ids:
                results[sid][model_name] = item.get("generated_caption", "").strip()

    except Exception as e:
        pass

# Print structured output for parsing
for tid in target_ids:
    print(f"\n%%% ID {tid} %%%") # Marker for easy parsing
    for model in ["SPE_Qwen2-7b_v2", "Qwen2-7b", "Llama-8b", "SPE_Gemma-9b_v2"]: # Selecting key models
        cap = results[tid].get(model, "[MISSING]")
        clean_cap = cap.replace("\n", " ")[:400] # Truncate to avoid huge logs
        print(f"[{model}]: {clean_cap}")
