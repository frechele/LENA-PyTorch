import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List


class SEBlock(nn.Sequential):
    def __init__(self, n_filters: int, reduction: int):
        super(SEBlock, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_filters, n_filters // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters // reduction, n_filters, 1),
            nn.Sigmoid()
        )


class ResBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, n_filters: int, stride: int=1, downsample: Union[None, nn.Module]=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_filters, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv3 = nn.Conv2d(n_filters, n_filters * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters * self.expansion)

        self.se = SEBlock(n_filters * self.expansion, 16)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv1(x)
        path = self.bn1(path)
        path = F.relu(path, inplace=True)

        path = self.conv2(path)
        path = self.bn2(path)
        path = F.relu(path, inplace=True)

        path = self.conv3(path)
        path = self.bn3(path)

        path = path * self.se(path)

        if self.downsample:
            x = self.downsample(x)

        return F.relu(x + path)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, layers: List[int]):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.fc = nn.Linear(512 * ResBlock.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, n_filters: int, blocks: int, stride: int=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != n_filters * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, n_filters * ResBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(n_filters * ResBlock.expansion)
            )

        layers = []
        layers.append(ResBlock(self.in_channels, n_filters, stride, downsample))
        self.in_channels = n_filters * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, n_filters))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x)
        x = self.fc(x)

        return x

def senet50(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, [3, 4, 6, 3])
