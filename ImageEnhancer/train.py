import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

# Paths to the folders
defocused_blurred_folder = '/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/blur_dataset_scaled/defocused_blurred'
motion_blurred_folder = '/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/blur_dataset_scaled/motion_blurred'
sharp_folder = '/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/blur_dataset_scaled/sharp'

# Load images
def load_images(folder, img_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            images.append(img)
    return np.array(images)

# Load all datasets
defocused_images = load_images(defocused_blurred_folder)
motion_blurred_images = load_images(motion_blurred_folder)
sharp_images = load_images(sharp_folder)

# Combine the blurred datasets (defocused and motion blurred)
blurred_images = np.concatenate([defocused_images, motion_blurred_images])

# The sharp images will be the target
target_images = np.concatenate([sharp_images, sharp_images])

# Define a simple UNet model
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    
    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c5)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Build and train the model
model = build_unet()

# Train the model
model.fit(blurred_images, target_images, epochs=50, batch_size=16, validation_split=0.2)

# Save the model
model.save('image_enhancer_model.h5')

# Use the trained model to enhance an image
def enhance_image(img_path, model, img_size=(256, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    enhanced_img = model.predict(img)
    enhanced_img = np.squeeze(enhanced_img, axis=0)
    
    return (enhanced_img * 255).astype(np.uint8)

# Example usage:
enhanced_img = enhance_image('/Users/dillonmaltese/Documents/git/AI/ImageEnhancer/test.jpg', model)
cv2.imwrite('enhanced_image.jpg', enhanced_img)
