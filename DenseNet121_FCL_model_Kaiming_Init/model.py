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
        backbone=models.densenet121(pretrained=False)
        
        self.backbone=nn.Sequential(*list(backbone.children())[:-1])
        
        # a fully connected layer for the ouput

        self.fc=nn.Sequential(nn.Linear(1024, 1),nn.Sigmoid())
                                          
        # self.fc=nn.Sequential(nn.Linear(1024, 1),
                                          # nn.ReLU(),
                                          # nn.Dropout(0.4),
                                          # nn.Linear(256,1),
                                          # nn.Sigmoid())
        
    def forward(self,x):

        x=self.backbone(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        
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
