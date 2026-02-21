import json
import os
import glob
import sys

# Directory matches my previous successful run
json_dir = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/json"

targets = {
    "Speech Bubble": ["speech bubble"],  # Relaxed from ["speech bubble", "minimalist"]
    "Cube": ["cube", "three-dimensional"],
    "Human/Chinese": ["human figure"] 
}

results = {k: {} for k in targets}
gt_text = {k: "N/A" for k in targets}

files = glob.glob(os.path.join(json_dir, "*.json"))

print(f"Scanning {len(files)} files: {', '.join([os.path.basename(f) for f in files])}")

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
            gt = item.get("ground_truth", "").lower()
            caption = item.get("generated_caption", "").strip()
            
            for target_name, keywords in targets.items():
                if any(k in gt for k in keywords):
                    results[target_name][model_name] = caption
                    # Save one example of GT for verification
                    if gt_text[target_name] == "N/A":
                        gt_text[target_name] = gt[:100] + "..."

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# Print Results
for target, models in results.items():
    print(f"\n=== {target} ===")
    print(f"GT: {gt_text[target]}")
    # Sort models to keep consistent order (e.g. BLIP, Florence, Gemma, Qwen, SPE)
    for model, cap in sorted(models.items()):
        # Truncate for terminal display
        clean_cap = cap.replace("\n", " ").strip()
        if not clean_cap: clean_cap = "[EMPTY]"
        print(f"[{model}]: {clean_cap[:300]}")
