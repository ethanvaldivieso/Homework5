#!/usr/bin/env python
# coding: utf-8

# In[126]:


import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy
import tqdm
import pylidc
from torch.autograd import Variable


# In[127]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[128]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[129]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[130]:


w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(()) 
t_p = model(t_u, w1, w2, b)
t_p


# In[131]:


loss = loss_fn(t_p, t_c)
loss


# In[132]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w1 + delta, w2 + delta, b), t_c) -
loss_fn(model(t_u, w1- delta, w2 - delta, b), t_c)) / (2.0 * delta)


# In[133]:


learning_rate = 1e-4
w = w - learning_rate * loss_rate_of_change_w


# In[134]:


loss_rate_of_change_b = (loss_fn(model(t_u, w1, w2, b + delta), t_c) -
loss_fn(model(t_u, w1, w2, b - delta), t_c)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b


# In[135]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[136]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[137]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[138]:


def dmodel_dw(t_u, w1, w2, b):
    return t_u


# In[139]:


def dmodel_db(t_u, w1, w2, b):
    return 1.0


# In[140]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(), dloss_dw2.sum(), dloss_db.sum()])


# In[141]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w1, w2, b = params
        
        t_p = model(t_u, w1, w2, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w1, w2, b)
        
        params = params - learning_rate * grad
        
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params


# In[149]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[150]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[151]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[152]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[ ]:




