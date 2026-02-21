import json
import os

filepath = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/json/SPE_Qwen2-7b_v2_results.json"

try:
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    items = data.get("results", [])
    
    print(f"Found {len(items)} items in {os.path.basename(filepath)}\n")
    
    for item in items:
        sid = item.get("sample_id")
        gt = item.get("ground_truth", "")[:100]
        cap = item.get("generated_caption", "")[:150].replace("\n", " ")
        print(f"ID {sid} | GT: {gt}... | SPE: {cap}...")

except Exception as e:
    print(e)
