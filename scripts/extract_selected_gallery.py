import re
import os
import base64

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_file", default="/Users/emanuelediluzio/Desktop/università/tesi_magistrale/multi_model_comparison.html")
    parser.add_argument("--output_dir", default="/Users/emanuelediluzio/Desktop/università/tesi_magistrale/images/chapter5")
    parser.add_argument("--sample_ids", type=str, default="21")
    return parser.parse_args()

args = get_args()
html_path = args.html_file
output_dir = args.output_dir
target_ids = [int(x) for x in args.sample_ids.split(",")]

def extract_images():
    if not os.path.exists(html_path):
        print(f"File not found: {html_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"=== EXTRACTING SUCCESS IMAGES: {target_ids} ===")

    for tid in target_ids:
        # Try both quote styles
        marker_double = f'id="sample-{tid}"'
        marker_single = f"id='sample-{tid}'"
        
        if marker_double in content:
            marker = marker_double
        elif marker_single in content:
            marker = marker_single
        else:
            print(f"\n[Sample {tid}] NOT FOUND in HTML")
            continue
            
        parts = content.split(marker)
        if len(parts) < 2: 
            continue
            
        # Robust block splitting
        if "class='sample'" in parts[1]:
             sample_block = parts[1].split("class='sample'")[0]
        else:
             sample_block = parts[1].split('class="sample"')[0]
        
        # Regex to match src with single OR double quotes
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', sample_block)
        if img_match:
            src_data = img_match.group(1)
            if "data:image/png;base64," in src_data:
                base64_data = src_data.split("data:image/png;base64,")[1]
                try:
                    img_bytes = base64.b64decode(base64_data)
                    file_name = f"sample_{tid}.png"
                    file_path = os.path.join(output_dir, file_name)
                    
                    with open(file_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    print(f"[Sample {tid}] Saved to {file_path}")
                except Exception as e:
                    print(f"[Sample {tid}] Error decoding/saving: {e}")
            else:
                print(f"[Sample {tid}] Image source is not base64 PNG.")
        else:
             print(f"[Sample {tid}] Image tag not found.")

if __name__ == "__main__":
    extract_images()
