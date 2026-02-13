import re
import os
import json

html_path = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
target_ids = [9, 11, 13, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27]

def parse_targets():
    if not os.path.exists(html_path):
        print(f"File not found: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"=== ANALYZING TARGET SAMPLES: {target_ids} ===")

    for tid in target_ids:
        marker = f'id="sample-{tid}"'
        if marker not in content:
            continue
            
        parts = content.split(marker)
        if len(parts) < 2: 
            continue
            
        sample_block = parts[1].split('id="sample-')[0]
        
        print(f"\n--- Sample {tid} ---")
        
        # Extract Captions for key models
        models_to_check = ["Florence-2", "Qwen2-7b", "SPE+Qwen2-7b v2", "SPE+Gemma-9b v2"]
        
        caption_matches = re.findall(r'<td class="model-name">([^<]+)</td>.*?<div class="caption-text">(.*?)</div>', sample_block, re.DOTALL)
        
        captions = {m.strip(): c.replace("<br>", " ").strip() for m, c in caption_matches}
        
        for m in models_to_check:
            cap = captions.get(m, "MISSING")
            # Print enough to copy-paste, but handle newlines
            clean_cap = cap.replace("\n", " ").strip()
            print(f"[{m}]: {clean_cap[:200]}...") 

if __name__ == "__main__":
    parse_targets()
