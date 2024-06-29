import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x)

# Basic operations
y = torch.ones_like(x)
print(y)

# Addition
z = x + y
print(z)

# Matrix multiplication
w = torch.matmul(x, y.T)
print(w)

# Moving tensor to GPU (if available)
if torch.cuda.is_available():
    x = x.cuda()
    print(x)
