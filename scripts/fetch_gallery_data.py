import re
import os
import json

html_path = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
json_path = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/visual_benchmark_samples.json"
output_json = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/gallery_full_data.json"

# Adding 12 and 22
target_ids = [11, 13, 16, 17, 24, 25, 12, 22]

models_of_interest = [
    "Florence-2",
    "Qwen2-7b",
    "Llama-8b",
    "SPE+Qwen2-7b v2",
    "SPE+Gemma-9b v2"
]

def load_ground_truth():
    gts = {}
    if not os.path.exists(json_path):
        return gts
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("samples", []):
                sid = item.get("sample_id")
                if sid in target_ids:
                    gts[sid] = item.get("ground_truth_caption", "No GT found")
    except Exception:
        pass
    return gts

def parse_html_captions():
    if not os.path.exists(html_path):
        return {}

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    results = {}
    for tid in target_ids:
        results[tid] = {}
        marker = f'id="sample-{tid}"'
        if marker not in content:
            continue
            
        parts = content.split(marker)
        if len(parts) < 2: 
            continue
            
        sample_block = parts[1].split('id="sample-')[0]
        caption_matches = re.findall(r'<td class="model-name">([^<]+)</td>.*?<div class="caption-text">(.*?)</div>', sample_block, re.DOTALL)
        
        caption_map = {}
        for m, c in caption_matches:
            clean_c = c.replace("<br>", "\n").strip()
            clean_c = re.sub(r'\s+', ' ', clean_c) 
            caption_map[m.strip()] = clean_c
        
        for m in models_of_interest:
            results[tid][m] = caption_map.get(m, "MISSING")
            
    return results

def main():
    gts = load_ground_truth()
    model_caps = parse_html_captions()
    
    final_data = {}
    for tid in target_ids:
        final_data[tid] = {"GT": gts.get(tid, "N/A")}
        for m in models_of_interest:
            final_data[tid][m] = model_caps.get(tid, {}).get(m, "N/A")
            
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"Saved full data to {output_json}")

if __name__ == "__main__":
    main()
