import re
import os

html_path = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"

def parse_html_regex():
    if not os.path.exists(html_path):
        print(f"File not found: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the block for Sample 0
    # Structure: <div class="sample" id="sample-0"> ... </div>
    # But regex for nested divs is hard. simpler: split by id="sample-"
    
    parts = content.split('id="sample-0"')
    if len(parts) < 2:
        print("Sample 0 not found.")
        return

    # The content for sample 0 starts after the split and goes until the next sample or end of div
    # We can look for the next id="sample-" or just grab a chunk
    sample_block = parts[1].split('id="sample-')[0]
    
    print("=== SAMPLE 0 ANALYSIS ===")
    
    # Extract Image Base64 length
    img_match = re.search(r'<img[^>]+src="([^"]+)"', sample_block)
    if img_match:
         print(f"Image found: Yes (src length: {len(img_match.group(1))})")
    else:
         print("Image found: No")

    # Extract Captions
    # Structure: <td class="model-name">Name</td><td><div class="caption-text">Caption</div></td>
    # Regex: <td class="model-name">([^<]+)</td>\s*<td>\s*<div class="caption-text">([^<]+)</div>
    
    caption_matches = re.findall(r'<td class="model-name">([^<]+)</td>.*?<div class="caption-text">(.*?)</div>', sample_block, re.DOTALL)
    
    for model_name, caption_html in caption_matches:
        # Simple cleanup of HTML entities if needed, but mostly just text
        clean_caption = caption_html.replace("<br>", "\n").strip()
        print(f"\n[{model_name}]:\n{clean_caption[:400]}...")

if __name__ == "__main__":
    parse_html_regex()
