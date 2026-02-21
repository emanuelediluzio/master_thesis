from PIL import Image
import os

def process_image(input_path, output_path, tolerance=30):
    print(f"Processing {input_path}...")
    try:
        # Open and convert to RGBA
        img = Image.open(input_path).convert("RGBA")
        datas = img.getdata()
        
        # Get background color from top-left
        bg_color = datas[0]
        print(f"Detected background color: {bg_color}")
        
        newData = []
        for item in datas:
            # Check similarity (ignoring alpha for comparison)
            if all(abs(item[i] - bg_color[i]) < tolerance for i in range(3)):
                # Replace with white
                newData.append((255, 255, 255, 255))
            else:
                newData.append(item)
        
        img.putdata(newData)
        img.save(output_path, "PNG")
        print(f"Saved processed image to {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    input_file = "/Users/emanuelediluzio/Desktop/2F381CE7D58EA33A8C96483562A3A136.jpg"
    output_file = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures/encoder_decoder_new.png"
    
    process_image(input_file, output_file)
