import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        n_inputs = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False
            # Replace the last fully-connected layer
        self.model.fc = nn.Sequential(nn.Linear(n_inputs, 512),nn.Dropout(0.5), nn.Linear(512, 512), nn.Dropout(0.5),nn.Linear(512, nclasses))
    
    def forward(self, x):
        x = self.model(x)
        return x
