# model of VGG16 with fully connected layers as head

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
        
        self.backbone=models.vgg16(pretrained=False)
        
        self.backbone.classifier[6]=nn.Sequential(nn.Linear(4096, 2))
                                          
        # self.backbone=nn.Sequential(*list(self.backbone.children())[:-1])
        
        # self.fc=nn.Sequential(nn.Linear(4096, 256),
                                # nn.ReLU(),
                                # nn.Dropout(0.4),
                                # nn.Linear(256,2))
        
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