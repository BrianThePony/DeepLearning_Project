import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import network
import utils
import pyttsx3
from engine import train_one_epoch, evaluate



# Hyper Parameters
num_classes = 2

# Load Datasets
import loadData
data = loadData.Dataset('',None)

dataset = data
dataset_test = data

# Split datasets into train/test
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


# %%
# Get network
model = network.get_model(num_classes)


# Split datasets into train/test
import random
import numpy as np
random.seed(1337)
np.random.seed(1337)

noIMG = 2177
train_IMG = []
train_Labels = [] 
test_IMG = []
test_Labels = []

a = list(range(2177))
b = (random.sample(a,int(len(a)*0.8)))
c = list(set(a) - set(b))

for i in b:
    #print(i)
    d = data.__getitem__(i)
    
    train_IMG.append(d[0])
    train_Labels.append(d[1])
for i in c:
    #print(i)
    d = data.__getitem__(i)
    
    test_IMG.append(d[0])
    test_Labels.append(d[1])



# Get network

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# Define learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# %%
# Model training
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=1,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)
for epoch in range(num_epochs):
    print("Epoch: %d\n",epoch)
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# Model Show
