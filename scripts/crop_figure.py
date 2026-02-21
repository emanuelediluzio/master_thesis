from PIL import Image

def crop_header(image_path, output_path, pixels_to_crop):
    try:
        img = Image.open(image_path)
        width, height = img.size
        # Crop the top 'pixels_to_crop' pixels
        # Box is (left, upper, right, lower)
        cropped_img = img.crop((0, pixels_to_crop, width, height))
        cropped_img.save(output_path)
        print(f"Successfully cropped {pixels_to_crop} pixels from top. New size: {cropped_img.size}")
    except Exception as e:
        print(f"Error cropping image: {e}")

if __name__ == "__main__":
    # Path to the image
    img_path = "figures/screenshot_3.png"
    # Overwrite the original
    crop_header(img_path, img_path, 80) # Cropping 80 pixels to be safe (approx 12% of 650)
