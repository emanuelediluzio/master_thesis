import json
import os
import glob

# Directory containing the result files
json_dir = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/json"

# Target concepts to find in Ground Truth
targets = {
    "Speech Bubble": ["speech bubble", "minimalist graphic"],
    "Cross/X": ["circle and a cross", "perpendicular lines"],
    "Chinese/Human": ["human figure", "stylized figure"] 
}

# Store results: {Target -> {Model -> Caption}}
results = {k: {} for k in targets}

files = glob.glob(os.path.join(json_dir, "*.json"))

print(f"Scanning {len(files)} files...")

for filepath in files:
    model_name = os.path.basename(filepath).replace("_results.json", "")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Handle different structures (list vs dict with 'results' key)
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "results" in data:
            items = data["results"]
            
        for item in items:
            gt = item.get("ground_truth", "").lower()
            caption = item.get("generated_caption", "").strip()
            
            for target_name, keywords in targets.items():
                # Check if ALL keywords match the Ground Truth
                if all(k in gt for k in keywords):
                    results[target_name][model_name] = caption

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# Print Results
for target, models in results.items():
    print(f"\n=== {target} ===")
    for model, cap in sorted(models.items()):
        # Truncate caption for readability if needed, or keep full
        clean_cap = cap.replace("\n", " ")[:200]
        if not clean_cap: clean_cap = "[EMPTY]"
        print(f"[{model}]: {clean_cap}")
