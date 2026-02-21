import json
import os
import glob

json_dir = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/json"
target_ids = [12, 38, 22]

files = glob.glob(os.path.join(json_dir, "*.json"))
results = {tid: {} for tid in target_ids}

print(f"Scanning {len(files)} files for IDs {target_ids}...")

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
            # Check 'sample_id' or 'id'
            sid = item.get("sample_id") or item.get("id")
            if sid in target_ids:
                results[sid][model_name] = item.get("generated_caption", "").strip()

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# Print Results
for tid in target_ids:
    print(f"\n=== Sample ID {tid} ===")
    for model, cap in sorted(results[tid].items()):
        clean_cap = cap.replace("\n", " ").strip()
        if not clean_cap: clean_cap = "[EMPTY]"
        print(f"[{model}]: {clean_cap[:300]}")
