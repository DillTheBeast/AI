import os
import torch
from model import SimpleNN, load_data, train_model, test_model, load_model
import tkinter
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# Load the data
train_loader, test_loader = load_data()

model_path = 'model.pth'

def TrainOrTest():
    if os.path.exists(model_path):
        # Load the trained model
        model = load_model(model_path)
        print("Loaded the trained model from disk.")
    else:
        # Initialize the model
        model = SimpleNN()
        # Train the model
        train_model(model, train_loader, num_epochs=5)
        print("Trained a new model and saved it to disk.")
    return model

# Load or train the model
model = TrainOrTest()

def uploadPhoto():
    file_path = filedialog.askopenfilename(
        title="Select a Photo",
        filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("JPEG Files", "*.jpeg"), ("GIF Files", "*.gif"), ("BMP Files", "*.bmp")]
    )
    if file_path:
        # Process the image and predict
        predicted_digit = process_and_predict(file_path)
        messagebox.showinfo("Information", f"Selected file: {file_path}\nPredicted Digit: {predicted_digit}")
    else:
        messagebox.showwarning("Warning", "No file selected")

def process_and_predict(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    
    # Invert the colors (MNIST has black digits on white background)
    image = ImageOps.invert(image)
    
    # Resize the image to 28x28
    image = image.resize((28, 28))
    
    # Thresholding
    image = image.point(lambda p: p > 128 and 255)
    
    # Normalize the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Visualize the processed image (optional)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title("Resized and Normalized Image")
    plt.show()
    
    # Predict the digit
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("720x480")
app.title("Hieroglyphic Scanner")

title = ctk.CTkLabel(app, text="Welcome to the Hieroglyphic Scanner", text_color="white")
title.pack(padx=10, pady=10)

pictureButton = ctk.CTkButton(app, text="Upload a photo", command=uploadPhoto)
pictureButton.pack(padx=10, pady=10)

app.mainloop()
