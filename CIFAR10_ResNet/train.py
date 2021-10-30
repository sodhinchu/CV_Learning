#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn.functional as F
from tqdm import tqdm

import oncecycle


# In[4]:


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


# In[8]:

def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
        


def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom




def cifar10train(model, device, trainloader, optimizer, criterion, epochs=2, scheduler=None, use_cycle=False, send_cycle=None):
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
          
            if use_cycle:    
                lr, mom = send_cycle.calc()
                update_lr(optimizer, lr)
                update_mom(optimizer, mom)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    return model


# In[ ]:




