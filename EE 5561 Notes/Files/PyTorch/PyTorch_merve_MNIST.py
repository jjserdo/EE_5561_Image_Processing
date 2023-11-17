# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:59:04 2023

@author: justi
Merve lecture using PyTorch
"""

# %% Import Libraries

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# %% Load the MNIST Dataset

data_full = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)

# visualize the data
plt.imshow(data_full.data[0], cmap="gray")

# access the data 
data1 = data_full.data[120]
plt.imshow(data1, cmap="gray")

# visualize multiple images
figure = plt.figure(figsize=(10,10))
cols, rows = 5, 5
for i in range(1, cols*rows+1):
    idx = torch.randint(0,len(data_full),(1,))
    
    img, label = data_full[idx]
    
    figure.add_subplot(rows, cols, i)
    plt.title('Number: ' + str(label))
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')

# JJ Code
"""
fig, ax = plt.subplots(5,5,figsize=(10,10))
for i in range(5):
    for j in range(5):
        idx = torch.randint(len(data_full))
        ax[i,j].imshow(data_full.data[idx])
"""

# %% Split the Dataset

# train, test, validation separation
train_data, test_data, valid_data = torch.utils.data.random_split(data_full, (30000, 10000, 20000))

batch_size = 100

loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True),
           'test':  torch.utils.data.DataLoader(test_data,  batch_size = batch_size, shuffle = True),
           'valid': torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = True)}

# visualize the dictiionary
train_part = loaders.get('train')
data2 = train_part.dataset
element1 = data2
element1 = data2[0][0].squeeze()
plt.imshow(element1, cmap='gray')


# %% LeNet

class LeNet(nn.Module): # This is from the lecture
    def __init__(self):
        super(LeNet.self).__init()
        self.conv1 = nn.Conv2d(1,6,5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(1,6,5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(256,120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120,84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84,10)
        self.relu5 = nn.ReLU()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
    

# %% Setting up the Model
device = torch.device('cuda')
model = LeNet()
model.to(device)

# define the loss function
criterion = nn.CrossEntropyLoss()

# define the optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD

num_epochs = 10
loss_list = []
loss_list_mean = []

# %% Train the model

for epoch in range(num_epochs)
    loss_list = 
    iter += 1
    if iter % 10 == 0:
        print(f'Iterations: {iter}')

# %% Validation

    if iter % 100 == 0:
        # accuracy
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(loaders):
            # getting the imabes and labels from the loaders
            images = images.to(device)
            labels = labels.to(device)
            
            # clear the gradients
            optimizer.zero_grad()
            
            # call the NN
            outputs = model(images)
            
            # get the predictions
            _, predicted = torch.max(outputs.data, 1)
            torch += label.size(0)
            correct += (predicted == labels).sum()
        
        accuracy = 100 * correct/total
        loss_list_mean = loss_list_mean.append(loss.item())
        
# visualize the loss
plt.plot(loss_list)
plt.plot(loss_list_mean)

# %% Test

    