"""Holds all model classes

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class gradcam_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(6400, 128, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return x

    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
def resnet_model_modified():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.fc = nn.Linear(in_features=64, out_features=10, bias=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    model.layer1 = nn.Sequential(
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer2 = Identity()
    model.layer3 = Identity()
    model.layer4 = Identity()
    return model


class MNIST_model(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(MNIST_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1  , out_channels=32 , kernel_size=3, stride=1), nn.BatchNorm2d(32) , nn.ReLU(),
            nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=3, stride=1), nn.BatchNorm2d(48) , nn.ReLU(),
            nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=3, stride=1), nn.BatchNorm2d(64) , nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels=80 , kernel_size=3, stride=1), nn.BatchNorm2d(80) , nn.ReLU(),
            nn.Conv2d(in_channels=80 , out_channels=96 , kernel_size=3, stride=1), nn.BatchNorm2d(96) , nn.ReLU(),
            nn.Conv2d(in_channels=96 , out_channels=112, kernel_size=3, stride=1), nn.BatchNorm2d(112), nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3, stride=1), nn.BatchNorm2d(144), nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3, stride=1), nn.BatchNorm2d(160), nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3, stride=1), nn.BatchNorm2d(176), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11264, n_classes),
            nn.BatchNorm1d(n_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        out = self.final_conv(x)
        return self.classifier(out)