
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class GeocellResNet(nn.Module):
    def __init__(self, num_classes):
        super(GeocellResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)