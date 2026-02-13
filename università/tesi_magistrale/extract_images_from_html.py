import os
import re
import base64

HTML_PATH = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
OUTPUT_DIR = "images/chapter5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_image_regex(content, sample_id, filename):
    print(f"Searching for Sample {sample_id}...")
    # Regex to find the Sample header and capture the following image src
    # Pattern looks for "Sample X" then lazily matches until <img src="data:image/png;base64,..."
    pattern = rf"Sample {sample_id}:.*?<img\s+[^>]*src=[\"']data:image/[a-zA-Z]+;base64,([^\"']+)[\"']"
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Warning: Could not find image data for Sample {sample_id}")
        return False
        
    base64_data = match.group(1)
    try:
        image_data = base64.b64decode(base64_data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Successfully saved {filename}")
        return True
    except Exception as e:
        print(f"Error decoding base64 for {sample_id}: {e}")
        return False

try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract ID 38 (Graduate)
    extract_image_regex(content, 38, "sample_38.png")

    # Extract ID 22 or 26
    if not extract_image_regex(content, 22, "sample_26.png"):
        print("Sample 22 not found, trying Sample 26 (Human)...")
        extract_image_regex(content, 26, "sample_26.png")

except FileNotFoundError:
    print(f"Error: HTML file not found at {HTML_PATH}")
