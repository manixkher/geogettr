
import torch.nn as nn
import torchvision.models as models


class GeocellResNet(nn.Module):
    def __init__(self, num_classes):
        super(GeocellResNet, self).__init__()
        # self.resnet = models.resnet50()  # No pre-trained weights
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify FC layer
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)