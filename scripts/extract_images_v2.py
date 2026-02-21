import os
import re
import base64

HTML_PATH = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
OUTPUT_DIR = "images/chapter5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_image_by_filename(content, target_filename, output_name):
    print(f"Searching for image corresponding to {target_filename}...")
    # Pattern: <img ... src="data...base64,DATA" ...> ... File: target_filename
    # We look for the base64 data closely followed by the filename div
    # Note: The grep output showed the img tag came BEFORE the file name div.
    
    # We'll regex for the img tag up to the filename
    # capturing the base64 string
    pattern = rf'<img\s+[^>]*src="data:image/[a-zA-Z]+;base64,([^"]+)"[^>]*>.*?File: {re.escape(target_filename)}'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        # Try scanning backwards? Or maybe just find the index of filename and look back?
        print(f"Regex failed for {target_filename}. Trying reverse search...")
        file_idx = content.find(f"File: {target_filename}")
        if file_idx == -1:
            print("File name not found in HTML.")
            return False
            
        # Search backwards for '<img'
        img_start = content.rfind("<img", 0, file_idx)
        if img_start == -1:
            print("No img tag found before file name.")
            return False
            
        img_tag = content[img_start:file_idx]
        src_match = re.search(r'src="data:image/[a-zA-Z]+;base64,([^"]+)"', img_tag)
        if not src_match:
            print("No base64 src found in img tag.")
            return False
            
        base64_data = src_match.group(1)
    else:
        base64_data = match.group(1)

    try:
        image_data = base64.b64decode(base64_data)
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Successfully saved {output_name}")
        return True
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return False

with open(HTML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Exact filename for ID 38 (Graduate) found in grep: sr_122237.svg
if extract_image_by_filename(content, "sr_122237.svg", "sample_38.png"):
    print("Extracted ID 38 (Graduate)")

# Exact filename for ID 22 (Human Silhouette) found in grep: wd_1229923.svg
if extract_image_by_filename(content, "wd_1229923.svg", "sample_22.png"):
    print("Extracted ID 22 (Human)")

