import torch
import torch.optim as optim
import torch.nn as nn
from main import get_model

# Instantiate the network
net = get_model()

# Dummy data
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
targets = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
