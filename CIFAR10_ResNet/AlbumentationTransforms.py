#!/usr/bin/env python
# coding: utf-8

# In[16]:


import albumentations as AL
import albumentations.pytorch as AT
import numpy as np


# In[14]:


def getAlbumentationTransform():
    transform = AL.Compose({
        #AL.Resize(200, 300),
        #AL.CenterCrop(100, 100),
        #AL.RandomCrop(32, 32),
        #AL.HorizontalFlip(p=0.5),
        #AL.Rotate(limit=(-90, 90)),
        #AL.VerticalFlip(p=0.5),
        AL.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))
        })
    return transform


# In[ ]:


'''
AL.Resize(200, 300),
        AL.CenterCrop(100, 100),
        AL.RandomCrop(80, 80),
        AL.HorizontalFlip(p=0.5),
        AL.Rotate(limit=(-90, 90)),
        AL.VerticalFlip(p=0.5),
        AL.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

'''


# In[17]:


class AlbumentationTransform:
    def __init__(self, transforms_list=[]):    
        #transforms_list.append(AT.ToTensor())
        #self.transforms_new = AL.Compose(transforms_list)
        self.aug = AL.Compose(transforms_list)
        
    def __call__(self,img):
        img = np.array(img)
        #img = self.transforms_new(image = img)['image']
        img = self.aug(image = img)['image']
        return img


# In[ ]:




