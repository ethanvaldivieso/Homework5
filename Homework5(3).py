#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[ ]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[ ]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[ ]:


w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(()) 
t_p = model(t_u, w1, w2, b)
t_p


# In[ ]:


loss = loss_fn(t_p, t_c)
loss


# In[ ]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w1 + delta, w2 + delta, b), t_c) -
loss_fn(model(t_u, w1- delta, w2 - delta, b), t_c)) / (2.0 * delta)


# In[ ]:


learning_rate = 1e-4
w1 = w1 - learning_rate * loss_rate_of_change_w


# In[ ]:


loss_rate_of_change_b = (loss_fn(model(t_u, w1, w2, b + delta), t_c) -
loss_fn(model(t_u, w1, w2, b - delta), t_c)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b


# In[ ]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[ ]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[ ]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[ ]:


def dmodel_dw(t_u, w1, w2, b):
    return t_u


# In[ ]:


def dmodel_db(t_u, w1, w2, b):
    return 1.0


# In[ ]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(), dloss_dw2.sum(), dloss_db.sum()])


# In[ ]:


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


# In[ ]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[ ]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[ ]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[ ]:


t_un = 0.1 * t_u
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[ ]:


Housing = pd.read_csv('Housing.csv')
Housing.head()


# In[ ]:


X = Housing[['area', 'bedrooms', 'bathrooms','stories', 'parking']]
y = Housing['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size=0.2, random_state=101)


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[ ]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


class DataMaker(Data.Dataset):
    def __init__(self, X, y):
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        self.targets = scaler.fit_transform(X.astype(np.float32))
        self.labels = y.astype(np.float32)
    
    def __getitem__(self, i):
        return self.targets[i, :], self.labels[i]

    def __len__(self):
        return len(self.targets)


# In[ ]:


class Model(nn.Module):
    def __init__(self, n_features, hiddenA, hiddenB):
        super(Model, self).__init__()
        self.linearA = nn.Linear(n_features, hiddenA)
        self.linearB = nn.Linear(hiddenA, hiddenB)
        self.linearC = nn.Linear(hiddenB, 1)

    def forward(self, x):
        yA = F.relu(self.linearA(x))
        yB = F.relu(self.linearB(yA))
        return self.linearC(yB)


# In[ ]:


torch.manual_seed(1)


# In[ ]:


train_set = DataMaker(X_train, y_train)
test_set = DataMaker(X_test, y_test)


# In[ ]:


bs = 25
train_loader = Data.DataLoader(train_set, batch_size=bs, shuffle=True)
test_loader = Data.DataLoader(test_set, batch_size=bs, shuffle=True)


# In[ ]:


net = Model(n_features, 100, 50)


# In[ ]:


criterion = nn.MSELoss(size_average=False)
optimizer = optim.Adam(net.parameters(), lr=0.01)


# In[ ]:


n_epochs = 200
all_losses = []
for epoch in range(n_epochs):
    progress_bar = tqdm.notebook.tqdm(train_loader, leave=False)
    losses = []
    total = 0
    for inputs, target in progress_bar:
        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(y_pred, torch.unsqueeze(target,dim=1))

        loss.backward()
        
        optimizer.step()
        
        progress_bar.set_description(f'Loss: {loss.item():.3f}')
        
        losses.append(loss.item())
        total += 1

    epoch_loss = sum(losses) / total
    all_losses.append(epoch_loss)
                
    mess = f"Epoch #{epoch+1}\tLoss: {all_losses[-1]:.3f}"
    tqdm.tqdm.write(mess)


# In[ ]:


plt.plot(all_losses)


# In[ ]:


y_pred = []
y_true = []
net.train(False)
for inputs, targets in test_loader:
    y_pred.extend(net(inputs).data.numpy())
    y_true.extend(targets.numpy())
plt.scatter(y_pred, y_true)
plt.plot([0, 50], [0, 50], '--k')


# In[ ]:


print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("R^2:", r2_score(y_true, y_pred))


# In[1]:


from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import QuantileTransformer

regr_trans = TransformedTargetRegressor(
    regressor=RidgeCV(),
    transformer=QuantileTransformer(
        n_quantiles=300, output_distribution='normal'))

regr_trans.fit(X_train, y_train)
y_pred = regr_trans.predict(X_test)
plt.scatter(y_pred, y_test)
plt.plot([0, 50], [0, 50], '--k')
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

