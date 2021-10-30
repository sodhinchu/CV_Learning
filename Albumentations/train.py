#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch.nn.functional as F
from tqdm import tqdm
import torch


# In[10]:


def train(model, device, train_loader, optimizer, l1reg=False):
    model.train()
    train_loss = 0
    correct = 0
    processed = 0
    train_losses = []
    train_accuracy = []

    if(l1reg):
        print('L1 Reg Enabled')
    else:
        print('L1 Reg Not Enabled')
    
    for param in optimizer.param_groups:
        if(param['weight_decay']):
            print('L2 Reg Enabled')
        else:
            print('L2 Reg Not Enabled')

    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if(l1reg):
            loss += l1_regularizer.calcL1Loss(model)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        #train_losses.append(loss)
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')

    train_loss /= len(train_loader.dataset)
    accuracy = 100*correct/len(train_loader.dataset)
    train_accuracy.append(100*correct/len(train_loader.dataset))
    train_losses.append(train_loss)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    #Also return the final model
    return train_loss, accuracy


# In[11]:


def cifar10train(model, device, trainloader, optimizer, criterion, pbar):
    model.train()    
    train_losses = []
    train_loss = 0
    train_accuracy = []
    correct = 0
    processed = 0
    #pbar = tqdm(trainloader)

    #for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    train_loss = 0
    correct = 0
    processed = 0        
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # sum up batch loss
        _, predicted = torch.max(outputs.data, 1)  # get the index of the max log-probability
        correct += (predicted == labels).sum().item()
        processed += labels.size(0)            
        pbar.set_description(desc= f'loss={loss.item()} batch_id={i} Accuracy = {100*correct/processed:0.2f}')          
                               
    train_loss /= len(trainloader.dataset)
    accuracy = 100*correct/len(trainloader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(trainloader.dataset),accuracy))                                            

    return accuracy, train_loss


# In[ ]:




