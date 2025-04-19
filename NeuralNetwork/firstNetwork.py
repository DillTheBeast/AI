import tensorflow as tf
import numpy as np

# Generate Data
x = np.array(range(1, 100), dtype=np.float32)
y = np.sqrt(x)

# Normalize x
x = (x - x.min()) / (x.max() - x.min())  # Scale x to [0,1]

# Train/Test Split
trainSize = int(len(x) * 0.8)
xTrain, xTest = x[:trainSize], x[trainSize:]
yTrain, yTest = y[:trainSize], y[trainSize:]

# Ensure NumPy format (avoid retracing issues)
xTrain = np.array(xTrain, dtype=np.float32).reshape(-1, 1)
xTest = np.array(xTest, dtype=np.float32).reshape(-1, 1)
yTrain = np.array(yTrain, dtype=np.float32)

# Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Explicit Input layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model (Fewer Epochs & Verbose Output)
model.fit(xTrain, yTrain, epochs=500, batch_size=10, verbose=1)

# Predict on Test Data
predictions = model.predict(xTest)

# Print Results
print("xTest:", xTest.flatten())
print("Predictions on xTest:", predictions.flatten())