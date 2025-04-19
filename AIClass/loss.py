import torch
import torch.nn as nn

#Define a neural network class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        #Define a fully connected layer with 2 input features and 1 output feature
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        #The forward function applies the linear layer on the input x
        return self.fc(x)
    
def verify(w1, w2, b):
    x1 = (w1 * 1)
    x2 = (w2 * 2)
    return x1 + x2 + b
#instantiate the model
model = SimpleModel()

model.fc.weight = nn.Parameter(torch.tensor([[0.5, -0.5]]))

#Create a sample input tensor of size (batch_size=1m input_features=2, means 2 arguments or 2 inputs)
input_data = torch.tensor([[1.0, 2.0]])

#Pass the input through the model
output = model(input_data)
target = output
criterion = nn.CrossEntropyLoss()
loss = criterion(output, output)

print(output)
print(loss)