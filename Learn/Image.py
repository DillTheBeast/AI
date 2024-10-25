from PIL import Image, ImageEnhance
import cv2
import numpy as np

# Load the image using PIL
imagePath = '/Users/dillonmaltese/Documents/git/AI/Learn/images.jpeg'
image = Image.open(imagePath)

# Convert PIL image to OpenCV format (numpy array)
# image_cv = np.array(image)

# # Convert to grayscale for blurriness detection
# gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

# # Compute the Laplacian variance to check blurriness
# laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# # Define a threshold for blurriness (can be adjusted)
# blurriness_threshold = 100  # Lower values indicate higher blur

# if laplacian_var < blurriness_threshold:
#     print("The image is blurry.")
# else:
#     print("The image is not blurry.")

# # Enhance the brightness using PIL
# new = ImageEnhance.Brightness(image)
# factor = 2
# enhanced = new.enhance(factor)

# # Show the enhanced image
# enhanced.show()
# image.show()
pixels = image.load()

image.line((0, 0), (image.size[0], image.size[1]))
            
image.show()

