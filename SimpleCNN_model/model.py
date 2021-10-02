# model of simple CNN

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # this is the same as nn.Module.__init__(self) 
        
        self.cnn=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.fc1=nn.Linear(32*6*6,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,1)
        self.classifier=nn.Sigmoid()
        
    def forward(self,x):

        x=self.cnn(x)
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.classifier(x)

        return x
