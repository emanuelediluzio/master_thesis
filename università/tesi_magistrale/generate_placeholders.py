from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(filename, text, size=(800, 400), color=(200, 200, 200)):
    img = Image.new('RGB', size, color=color)
    d = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        # This path is common on macOS
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()
        
    # Calculate text position (approximate centering)
    # text_width = d.textlength(text, font=font) # Newer Pillow
    # text_bbox = d.textbbox((0, 0), text, font=font)
    # text_width = text_bbox[2] - text_bbox[0]
    # text_height = text_bbox[3] - text_bbox[1]
    
    # Simple centering for older Pillow versions if needed, or just hardcode
    d.text((50, 180), text, fill=(0, 0, 0), font=font)
    
    output_path = os.path.join("/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures", filename)
    img.save(output_path)
    print(f"Created {output_path}")

placeholders = [
    "placeholder_raster_vs_vector.png",
    "placeholder_blip2_architecture.png",
    "placeholder_spectral_bias.png",
    "placeholder_path_simplification.png",
    "placeholder_hybrid_sequence.png",
    "placeholder_spe_detailed.png"
]

os.makedirs("/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures", exist_ok=True)

for p in placeholders:
    create_placeholder(p, f"PLACEHOLDER:\n{p}")
