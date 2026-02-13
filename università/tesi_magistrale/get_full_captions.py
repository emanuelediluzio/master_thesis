
import re

file_path = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
target_samples = ["sample-0", "sample-4", "sample-26", "sample-28", "sample-48", "sample-56"]

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    with open("captions.txt", "w", encoding="utf-8") as out:
        for sample_id in target_samples:
            out.write(f"=== {sample_id} ===\n")
            
            # Find start of sample
            start_marker = f'<div class="sample" id="{sample_id}">'
            start_idx = content.find(start_marker)
            if start_idx == -1:
                out.write("NOT FOUND\n")
                continue
                
            # Find end of sample (start of next sample or end of file)
            chunk = content[start_idx:]
            next_sample_match = re.search(r'<div class="sample" id="sample-', chunk[1:])
            if next_sample_match:
                chunk = chunk[:next_sample_match.start() + 1]
                
            # Extract Ground Truth
            gt_match = re.search(r'<div class="ground-truth">\s*Ground truth:\s*(.*?)\s*</div>', chunk, re.DOTALL)
            if gt_match:
                out.write(f"GT: {gt_match.group(1).strip()}\n")
            else:
                gt_match_lax = re.search(r'<div class="ground-truth">\s*(.*?)\s*</div>', chunk, re.DOTALL)
                if gt_match_lax:
                     out.write(f"GT: {gt_match_lax.group(1).replace('Ground truth:', '').strip()}\n")

            # Extract Models
            row_pattern = re.compile(r'<td class="model-name">\s*(.*?)\s*</td>\s*<td>\s*<div class="caption-text">\s*(.*?)\s*</div>\s*</td>', re.DOTALL)
            rows = row_pattern.findall(chunk)
            for name, text in rows:
                out.write(f"MODEL[{name.strip()}]: {text.strip()}\n")
            out.write("\n")

except Exception as e:
    print(f"Error: {e}")
