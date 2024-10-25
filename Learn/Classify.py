import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def classification(imgPath):
    # Load the input image
    img = cv2.imread(imgPath)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to read the image {imgPath}")
        return

    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    # Convert image array to the appropriate format
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    # Decode and display predictions
    print('Predicted:', decode_predictions(preds, top=3)[0][0][1])
    print('Predicted:', decode_predictions(preds, top=3)[0])

# Set the path where the images are stored
image_dir = "/Users/dillonmaltese/Documents/git/AI/Learn"
files = os.listdir(image_dir)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Loop through all the image files
for file in files:
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        # Create the full file path
        file_path = os.path.join(image_dir, file)
        classification(file_path)

# Optionally, classify a specific image
# classification("/Users/dillonmaltese/Documents/git/AI/Learn/test.jpg")
