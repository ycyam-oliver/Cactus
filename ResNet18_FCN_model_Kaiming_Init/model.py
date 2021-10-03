# ResNet model with fully convoluted network head

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # this is the same as nn.Module.__init__(self) 
        
        # gonna train every layer
        backbone=models.resnet18(pretrained=False)
        
        self.backbone=nn.Sequential(*list(backbone.children())[:-2])
        
        # use FC as head
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=100,kernel_size=1),
            nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100,out_channels=1,kernel_size=1)
        )
        
        self.classifier=nn.Sigmoid()
        
    def forward(self,x):

        x=self.backbone(x)
        x=self.fc(x)
        x=x.view(-1,1)
        x=self.classifier(x)
        
        return x

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)