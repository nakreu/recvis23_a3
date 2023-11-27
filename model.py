#Model 1
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
        #replace last-layer with an adapted classifier with nn.Dropout for regularization
        self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(n_inputs, nclasses))
    
    def forward(self, x):
        x = self.model(x)
        return x
