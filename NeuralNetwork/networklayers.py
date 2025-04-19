import torch
import torch.nn as nn
import numpy

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = torch.sqrt
        self.fc2 = torch.square

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x
    
net = MyNet()

input = torch.randn(1, 10).abs()
print("input:", input)

output = net(input)
print("output:", output)