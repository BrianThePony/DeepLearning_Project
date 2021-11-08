import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import network



# Hyper Parameters
num_classes = 2
num_infeatures = 10 #INSERT

# Load Datasets
import loadData
data = loadData.Dataset('',None)





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
    print(i)
    train_IMG.append(data.__getitem__(i))
    train_Labels.append(data.__getitem__(i))
for i in c:
    print(i)
    test_IMG.append(data.__getitem__(i))
    test_Labels.append(data.__getitem__(i))



# Get network

# Define optimizer



# Model training

# Model Show
