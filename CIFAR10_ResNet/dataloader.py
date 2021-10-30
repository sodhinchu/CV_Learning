#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# In[2]:


def MNISTDataloader(num_workers=1, batch_size=32, saveTo='../data', train_transform=None, test_transform=None):
    if(train_transform == None):
        train_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

    if(test_transform == None):
        test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])         

    trainData = datasets.MNIST(saveTo, transform=train_transform, train=True, download=True)
    testData = datasets.MNIST(saveTo, transform=test_transform, train=False)  
    trainDL = DataLoader(trainData, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    testDL = DataLoader(testData, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    return trainDL, testDL


# In[3]:


def CIFAR10Dataloader(num_workers=1, batch_size=4, saveTo='../data', train_transform=None, test_transform=None):
    if(train_transform == None):
        train_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))
                ])
    if(test_transform==None):
        test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))
                ])     
    trainData = datasets.CIFAR10(saveTo, transform=train_transform, train=True, download=True)
    testData = datasets.CIFAR10(saveTo, transform=test_transform, train=False)  
    trainDL = DataLoader(trainData, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    testDL = DataLoader(testData, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    return trainDL, testDL


# In[4]:


def getDataLoader(dataset, batch_size=4, train_transform=None, test_transform=None):
    if(dataset == 'CIFAR10'):
        return CIFAR10Dataloader(batch_size=batch_size, train_transform=train_transform, test_transform=test_transform)
    elif(dataset == 'MNIST'):
        return MNISTDataloader(batch_size=batch_size, train_transform=train_transform, test_transform=test_transform)


# In[ ]:




