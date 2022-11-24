#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy
import tqdm
import pylidc
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[59]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[60]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[61]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[62]:


w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(()) 
t_p = model(t_u, w1, w2, b)
t_p


# In[63]:


loss = loss_fn(t_p, t_c)
loss


# In[64]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w1 + delta, w2 + delta, b), t_c) -
loss_fn(model(t_u, w1- delta, w2 - delta, b), t_c)) / (2.0 * delta)


# In[65]:


learning_rate = 1e-4
w1 = w1 - learning_rate * loss_rate_of_change_w


# In[66]:


loss_rate_of_change_b = (loss_fn(model(t_u, w1, w2, b + delta), t_c) -
loss_fn(model(t_u, w1, w2, b - delta), t_c)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b


# In[67]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[68]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[69]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[70]:


def dmodel_dw(t_u, w1, w2, b):
    return t_u


# In[71]:


def dmodel_db(t_u, w1, w2, b):
    return 1.0


# In[72]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(), dloss_dw2.sum(), dloss_db.sum()])


# In[73]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w1, w2, b = params
        
        t_p = model(t_u, w1, w2, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w1, w2, b)
        
        params = params - learning_rate * grad
        if epoch % 500 == 0 or epoch == 1: 
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params


# In[74]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[75]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[76]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[77]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[78]:


Housing = pd.read_csv('Housing.csv')
Housing.head()


# In[79]:


X = Housing[['area', 'bedrooms', 'bathrooms','stories', 'parking']]
y = Housing['price']


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[81]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[82]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[85]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[86]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

