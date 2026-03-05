import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class GradCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            5,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():

            self.model.conv1.weight[:, :3] = old_conv.weight
            self.model.conv1.weight[:, 3:] = old_conv.weight[:, :2]

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, 2)


    def forward(self, x):

        return self.model(x)