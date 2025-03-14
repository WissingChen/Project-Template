from torch import nn
from torchvision.models import resnet101


class ResNet101(nn.Module):
    def __init__(self, cfgs):
        super(ResNet101, self).__init__()
        num_classes = cfgs["model"]["num_classes"]
        self.model = resnet101(num_classes=num_classes)
        
    def forward(self, x):
        pred = self.model(x)
        return pred
