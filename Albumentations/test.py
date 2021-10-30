#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch.nn.functional as F
from tqdm import tqdm
import torch


# In[6]:


def cifar10test(model, device, testloader, criterion):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    test_loss = 0
    correct = 0
    processed = 0 
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)

            test_loss += loss.item()            
            
            _, predicted = torch.max(outputs.data, 1)
            processed += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(testloader.dataset)
    accuracy = (100 * correct / processed)
    #print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    print('Test set: Average loss: ',test_loss, " Accurcy: ",accuracy)
    return accuracy, test_loss


# In[ ]:




