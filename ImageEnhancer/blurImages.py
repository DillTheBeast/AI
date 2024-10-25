import cv2
import os
import numpy as np

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image, kernel_size=(25, 25), sigma=20):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    for i in range(7):
        blurred_image = cv2.GaussianBlur(blurred_image, kernel_size, sigma)
    return blurred_image

# Function to downsample the image (simulate lower resolution)
def downsample_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Upscale it back to original size to match the high-res image size
    return cv2.resize(resized_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

# Load high-quality images from a directory
high_quality_images_dir = "/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/GoodImages"
blurry_images_dir = "/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/BadImages"

if not os.path.exists(blurry_images_dir):
    os.makedirs(blurry_images_dir)

for image_name in os.listdir(high_quality_images_dir):
    if image_name.endswith((".jpg", ".png")):
        image_path = os.path.join(high_quality_images_dir, image_name)
        high_quality_image = cv2.imread(image_path)

        # Apply degradation (blur, downsampling, etc.)
        blurry_image = apply_gaussian_blur(high_quality_image)
        # Or use downsample_image to simulate lower resolution
        # blurry_image = downsample_image(high_quality_image)

        # Save blurry images alongside high-quality ones
        blurry_image_path = os.path.join(blurry_images_dir, f"blurry_{image_name}")
        cv2.imwrite(blurry_image_path, blurry_image)

print("Blurry images generated and saved.")
