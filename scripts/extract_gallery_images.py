import os
import re
import base64

# Configuration
HTML_FILE = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"
OUTPUT_DIR = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/images/gallery"
TARGET_SAMPLES = [0, 4, 26, 28, 48, 56]

def extract_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find sample IDs and their images
    # <div class="sample" id="sample-0">...<img alt="Sample 0" src="data:image/png;base64,..."
    
    print(f"Scanning {HTML_FILE}...")
    
    for sample_id in TARGET_SAMPLES:
        # Construct exact search pattern for this sample
        # We look for id="sample-X" then capture the img src nearby
        # This simple regex assumes the img tag follows the div id relatively closely
        pattern = r'id="sample-' + str(sample_id) + r'".*?<img[^>]+src="data:image/png;base64,([^"]+)"'
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
            b64_data = match.group(1)
            try:
                img_data = base64.b64decode(b64_data)
                out_path = os.path.join(OUTPUT_DIR, f"sample_{sample_id}.png")
                with open(out_path, 'wb') as out_f:
                    out_f.write(img_data)
                print(f"Saved Sample {sample_id} to {out_path}")
            except Exception as e:
                print(f"Error decoding/saving Sample {sample_id}: {e}")
        else:
            print(f"Warning: Could not find image data for Sample {sample_id}")

if __name__ == "__main__":
    extract_images()
