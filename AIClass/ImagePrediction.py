import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

root = tk.Tk()
root.title('Image Classification with ResNet50')
root.geometry('800x600')

# Text widget to display data and results
text = tk.Text(root, height=30, width=100)
text.grid(column=0, row=0)

def open_folder():
    # Open a folder dialog
    folder_path = fd.askdirectory()
    if folder_path:
        image_predictions = []
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            # Ensure that it's a file
            if os.path.isfile(filepath):
                try:
                    # Open the image file
                    img = Image.open(filepath)
                    # Convert the image to RGB if it's not
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize the image to 224x224 pixels as required by ResNet50
                    img = img.resize((224, 224))
                    # Convert the image to a numpy array
                    x = image.img_to_array(img)
                    # Add an extra dimension for the batch size
                    x = np.expand_dims(x, axis=0)
                    # Preprocess the image data
                    x = preprocess_input(x)
                    # Make predictions
                    preds = model.predict(x)
                    # Decode the predictions
                    decoded_preds = decode_predictions(preds, top=3)[0]
                    # Format the predictions
                    prediction_text = f"Predictions for {filename}:\n"
                    for i, pred in enumerate(decoded_preds):
                        prediction_text += f"  {i+1}. {pred[1]} ({pred[2]*100:.2f}%)\n"
                    prediction_text += "\n"
                    image_predictions.append(prediction_text)
                except (UnidentifiedImageError, IOError):
                    # If PIL cannot identify the file, it's not an image; skip it
                    continue
        # Display the predictions
        text.delete('1.0', tk.END)  # Clear previous content
        if image_predictions:
            for prediction in image_predictions:
                text.insert(tk.END, prediction)
        else:
            text.insert(tk.END, "No image files found or unable to process images in the selected folder.\n")

# Button to open folder and trigger the prediction
open_button = ttk.Button(
    root,
    text='Open a Folder',
    command=open_folder
)

open_button.grid(column=0, row=1, padx=10, pady=10)

root.mainloop()