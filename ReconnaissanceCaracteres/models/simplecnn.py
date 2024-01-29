import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    """
    Basic fully connected neural-network
    """

    crop = 28

    def __init__(self):
        hidden1 = 100
        hidden2 = 100
        super(MyModel, self).__init__()
        self.hidden1 = nn.Linear(self.crop * self.crop, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, 10)

    def forward(self, x):
        x = x.view(-1, self.crop * self.crop)  # flatten the images before using fully-connected layers
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.softmax(x, dim=0)
        return x

