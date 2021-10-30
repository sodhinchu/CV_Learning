#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# uses this new ResNet Architecture for Cifar10:
# 
#     PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
#     Layer1 -
#         X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
#         R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
#         Add(X, R1)
#     Layer 2 -
#         Conv 3x3 [256k]
#         MaxPooling2D
#         BN
#         ReLU
#     Layer 3 -
#         X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
#         R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
#         Add(X, R2)
#     MaxPooling with Kernel Size 4
#     FC Layer 
#     SoftMax
# 

# In[5]:


class CustomResNet(nn.Module):
    def __init__(self):    
        super(CustomResNet, self).__init__()
        # PrepLayer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 32 ((32-3+2*1/1)+1)
        self.layer1_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )# output_size = 16

        self.r1 = self._res_block(128, 128, 1) # output_size = 32

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )# output_size = 8

        self.layer3_x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )# output_size = 4    

        self.r2 = self._res_block(512, 512, 1) # output_size = 4   

        self.pool = nn.MaxPool2d(4,4) # output_size = 1
        self.fc = nn.Linear(512,10)


    def _res_block(self, in_channels, out_channels, padding):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1_x(x)
        r1 = self.r1(x)
        x = torch.add(x, r1)
        x = self.layer2(x)
        x = self.layer3_x(x)
        r2 = self.r2(x)
        x = torch.add(x, r2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)    


# In[ ]:




