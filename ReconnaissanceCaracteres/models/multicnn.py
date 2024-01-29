import torch.nn as nn

class CNN(nn.Module):
    crop = 28

    def __init__(self):
        super(CNN, self).__init__()

        #Convolution/RelU/MaxPooling layers definition
        self.conv1 = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=1) # 1 to 2 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 to 16x16 pixels
        self.conv2 = nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=1)  # 2 to 4 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 to 8x8 pixels
        self.conv3 = nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=1)  # 4 to 8 channels
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 to 4x4 pixels

        # dense layers definition
        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 4 * 4 * 4) # flatten the data
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

