import numpy as np
import torch as py 
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

a = np.array([-1, 2, 4, 3, 7])
b = py.tensor(a, dtype=py.float32)

print("relu:",relu(a))
print("relu with pytorch:", py.relu(b))
print("tanh:",tanh(a))
print("tanh with pytortch:", py.tanh(b))
print("sigmoid:",sigmoid(a))
print("sigmoid with pytorch:", py.sigmoid(b))
print("softmax:", softmax(a))
print("softmax with pytorch:", py.softmax(b, dim=0))