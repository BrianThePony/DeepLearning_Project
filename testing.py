# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:57:01 2021

@author: BrianThePony aka Tagmus

This is for testing the model for other pictures of cans

"""
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
imlist = []
root = r"C:\Users\NikolajBj\OneDrive - Danmarks Tekniske Universitet\DTU\7. semester\02456 - Deep Learning\DeepLearning_Project\testims"
for i in list((os.listdir(root))):
    im = Image.open(os.path.join(root,i)).convert('RGB')
    im = np.asarray(im)
    im = np.transpose(im)
    imlist.append(im)
#imlist = np.asarray(imlist)
#imlist = torch.FloatTensor(imlist)
model_path = r"C:\Users\NikolajBj\OneDrive - Danmarks Tekniske Universitet\DTU\7. semester\02456 - Deep Learning\DeepLearning_Project\modelAllpix"
model = torch.load(model_path)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(len(imlist)):
    im = torch.from_numpy(imlist[i])
    im = im/255
    
    pred = model([im.to(device)])
    
    im = Image.fromarray(im.mul(255).permute(1, 2, 0).byte().numpy())
    sz = pred[0]['boxes'].size()
        # Create figure and axes
    fig, ax = plt.subplots()
        
        # Display the image
    ax.imshow(im)
        
        # Create a Rectangle patch
        
        
        # Add the patch to the Axes
    color = ["red","green"]
    
    for i in range(sz[0]):
        rect = patches.Rectangle((pred[0]['boxes'][i,0], pred[0]['boxes'][i,1]), pred[0]['boxes'][i,2]-pred[0]['boxes'][i,0], pred[0]['boxes'][i,3]-pred[0]['boxes'][i,1], linewidth=2, edgecolor=color[pred[0]['labels'][i]], facecolor='none')
        ax.add_patch(rect)
    plt.show()
