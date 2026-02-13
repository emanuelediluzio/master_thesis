import os
import re
import base64

HTML_PATH = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
OUTPUT_DIR_CH5 = "images/chapter5"
OUTPUT_DIR_GAL = "images/gallery"
os.makedirs(OUTPUT_DIR_CH5, exist_ok=True)
os.makedirs(OUTPUT_DIR_GAL, exist_ok=True)

def extract_correct_image(content, target_filename, output_path):
    print(f"Searching for image corresponding to {target_filename}...")
    
    # Find position of the filename
    file_label = f"File: {target_filename}"
    file_idx = content.find(file_label)
    if file_idx == -1:
        print(f"Error: label {file_label} not found.")
        return False
        
    # Heuristic: The image should be within the last 200000 chars before the label
    search_window = content[max(0, file_idx - 200000):file_idx]
    
    # Find ALL src="data:image..." matches in this window
    matches = list(re.finditer(r'src="data:image/[a-zA-Z]+;base64,([^"]+)"', search_window))
    
    if not matches:
        print("No base64 image found preceding the label.")
        return False
        
    # Take the LAST match (closest to the label)
    last_match = matches[-1]
    base64_data = last_match.group(1)
    
    try:
        image_data = base64.b64decode(base64_data)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Successfully saved {output_path} (Size: {len(image_data)} bytes)")
        return True
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return False

with open(HTML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Exact filename for ID 38 (Graduate) -> Chapter 5
extract_correct_image(content, "sr_122237.svg", os.path.join(OUTPUT_DIR_CH5, "sample_38.png"))

# Exact filename for ID 4 (Credit Card) -> Gallery
extract_correct_image(content, "ki_0272820.svg", os.path.join(OUTPUT_DIR_GAL, "sample_4.png"))

# Exact filename for ID 22 (Human) -> Chapter 5
extract_correct_image(content, "wd_1229923.svg", os.path.join(OUTPUT_DIR_CH5, "sample_22.png"))

# Exact filename for "Sample 5" (Smartphone, Index 4) -> Gallery
extract_correct_image(content, "ki_0202123.svg", os.path.join(OUTPUT_DIR_GAL, "sample_5.png"))
