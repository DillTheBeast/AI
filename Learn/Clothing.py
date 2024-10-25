from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import cv2

# Load both MNIST (digits) and Fashion MNIST (clothing)
(mnist_images, mnist_labels) = mnist.load_data()
(fashion_images, fashion_labels) = fashion_mnist.load_data()

# Resize MNIST images (though they're already 28x28, but doing this for your request)
resized_images = []
for img in mnist_images:
    resized_images.append(cv2.resize(img, 28, 28))  # Resizing each image
mnist_images = np.array(resized_images)  # Convert the list back to a NumPy array

# Resize Fashion MNIST images (though they're already 28x28)
resized_images = []
for img in fashion_images:
    resized_images.append(cv2.resize(img, 28, 28))  # Resizing each image
fashion_images = np.array(resized_images)  # Convert the list back to a NumPy array


# We will label MNIST images as 0 (not clothing) and Fashion MNIST as 1 (clothing)
mnist_labels = np.zeros(mnist_labels.shape[0])
fashion_labels = np.ones(fashion_labels.shape[0])

# Combine the datasets
images = np.concatenate([mnist_images, fashion_images], axis=0)
labels = np.concatenate([mnist_labels, fashion_labels], axis=0)

# Shuffle the data
indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[784], activation='sigmoid')  # Binary classification
])

model.compile(optimizer='sgd', loss='mean_squared_error')
