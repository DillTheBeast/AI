import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x

# Create an instance of the network
def get_model():
    return SimpleNN()

# Add this to ensure that this script can be run standalone as well
if __name__ == "__main__":
    net = get_model()
    print(net)
