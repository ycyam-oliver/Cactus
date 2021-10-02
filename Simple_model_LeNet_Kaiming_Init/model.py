# simple LeNet model

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # this is the same as nn.Module.__init__(self) 
        
        # build a network resembling the classic LeNet network
        
        self.conv1=nn.Conv2d(3,6,5) 
        # 3 channels input for the 3 color channels
        # 6 output filters of size 5x5
        
        self.conv2=nn.Conv2d(6,16,5)
        
        # a series of fully connected layers
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,2)
        
    def forward(self,x):

        # conv -> activate -> pooling
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)

        # reshape
        x=x.view(x.size()[0],-1)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
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