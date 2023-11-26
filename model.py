import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2,stride=1, padding=1)
        self.conv4 = nn.Conv2d(256,256, kernel_size=2, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.linear6 = nn.Linear(256*3*3, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.linear7 = nn.Linear(512,512)
        self.dropout7 = nn.Dropout(p=0.5)
        self.linear8 = nn.Linear(512,250)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.relu(self.maxpool5(x))
        x = self.linear6(x)
        x = F.relu(self.dropout6(x))
        x = self.linear7(x)
        x = F.relu(self.dropout7(x))
        x = self.linear8(x)
        return x
