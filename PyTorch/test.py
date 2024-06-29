import torch

# Normalize the input data
features = torch.tensor([[1500, 3, 10], [1800, 4, 15], [2400, 5, 20]], dtype=torch.float32)
prices = torch.tensor([300, 400, 500], dtype=torch.float32)

# Mean and standard deviation for normalization
mean = features.mean(dim=0)
std = features.std(dim=0)

# Normalize features
features = (features - mean) / std

# Initialize weights and bias with small random values
weights = torch.randn(3, requires_grad=True) * 0.01
bias = torch.randn(1, requires_grad=True) * 0.01

# Define the model
def model(x):
    return torch.matmul(x, weights) + bias

# Define a simple mean squared error loss function
def loss_fn(predictions, targets):
    return ((predictions - targets) ** 2).mean()

# Training loop
learning_rate = 0.001
for epoch in range(100):
    # Make predictions
    predictions = model(features)
    
    # Calculate the loss
    loss = loss_fn(predictions, prices)
    
    # Backpropagate the gradients
    loss.backward()
    
    # Update weights and bias
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
        
        # Zero the gradients
        weights.grad.zero_()
        bias.grad.zero_()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# Test the model with normalized data
new_house = torch.tensor([2000, 4, 12], dtype=torch.float32)
new_house = (new_house - mean) / std
predicted_price = model(new_house)
print(f'Predicted price for the new house: ${predicted_price.item() * 1000:.2f}')
