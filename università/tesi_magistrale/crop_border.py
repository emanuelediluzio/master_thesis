from PIL import Image, ImageChops

def trim_border(input_path, output_path, border_pixels=2):
    print(f"Processing {input_path}...")
    try:
        img = Image.open(input_path).convert("RGBA")
        
        # Option 1: Simple crop if the border is a fixed width
        # width, height = img.size
        # img = img.crop((border_pixels, border_pixels, width - border_pixels, height - border_pixels))
        
        # Option 2: Auto-trim based on content (better if border is distinct from background)
        # Assuming background is white/transparent, and border is grey.
        # But user said "remove contour".
        # Let's try a simple crop first, as screenshot borders are usually at the very edge.
        
        width, height = img.size
        # Crop 3 pixels from each side to be safe
        img = img.crop((3, 3, width - 3, height - 3))
        
        img.save(output_path, "PNG")
        print(f"Saved cropped image to {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    input_file = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures/encoder_decoder_new.png"
    trim_border(input_file, input_file)
