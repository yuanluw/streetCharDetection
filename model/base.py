# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 12:59, matt '

import torch
import torch.nn as nn

from torchvision import models


class base_model(nn.Module):
    def __init__(self, verbose=False):
        super(base_model, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(*list(model_conv.children())[:-1])

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
        self.fc6 = nn.Linear(512, 11)
        self.verbose = verbose

    def forward(self, x):
        x = self.conv(x)
        if self.verbose:
            print("conv size: ", x.shape)
        x = x.view(x.shape[0], -1)
        if self.verbose:
            print("view size: ", x.shape)
        c1 = self.fc1(x)
        if self.verbose:
            print("fc size: ", c1.shape)
        c2 = self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)
        c5 = self.fc5(x)
        c6 = self.fc6(x)

        return c1, c2, c3, c4, c5, c6


class st_model(nn.Module):
    def __init__(self, verbose=False):
        super(st_model, self).__init__()

        model_conv = models.resnet101(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(*list(model_conv.children())[:-1])

        self.fc1 = nn.Linear(2048, 11)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 11)
        self.fc4 = nn.Linear(2048, 11)
        self.fc5 = nn.Linear(2048, 11)
        self.fc6 = nn.Linear(2048, 11)
        self.verbose = verbose

    def forward(self, x):
        x = self.conv(x)
        if self.verbose:
            print("conv size: ", x.shape)
        x = x.view(x.shape[0], -1)
        if self.verbose:
            print("view size: ", x.shape)
        c1 = self.fc1(x)
        if self.verbose:
            print("fc size: ", c1.shape)
        c2 = self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)
        c5 = self.fc5(x)
        c6 = self.fc6(x)

        return c1, c2, c3, c4, c5, c6


if __name__ == "__main__":
    x = torch.randn((1, 3, 64, 128))
    model = models.resnet50()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    print(model)
    y = model(x)
    print(x.shape)