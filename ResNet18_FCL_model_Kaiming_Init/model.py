# ResNet model with fully connected head

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
        self.backbone=models.resnet18(pretrained=False)

        # a fully connected layer for the ouput
        num_filters=self.backbone.fc.in_features

        self.backbone.fc=nn.Sequential(nn.Linear(num_filters, 1),
                                          nn.Sigmoid())
        
    def forward(self,x):

        x=self.backbone(x)
        
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