# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:02:29 2021

@author: silla

Test to track cans, from sequential images
"""
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
imlist = []
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
root = r"C:\Users\silla\OneDrive - Danmarks Tekniske Universitet\DTU\7. semester\02456 - Deep Learning\DeepLearning_Project\tracking"
for i in list((os.listdir(root))):
    im = Image.open(os.path.join(root,i)).convert('RGB')
    im = np.asarray(im)
    im = np.transpose(im,(2,0,1))#np.transpose(im)
    imlist.append(im)
#imlist = np.asarray(imlist)
#imlist = torch.FloatTensor(imlist)
model_path = r"C:\Users\silla\OneDrive - Danmarks Tekniske Universitet\DTU\7. semester\02456 - Deep Learning\DeepLearning_Project\modelAllpix"
pred = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for j in range(len(imlist)):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = torch.load(model_path,map_location=device)
    model.eval()
    im = torch.from_numpy(imlist[j])
    im = im/255
    tempPred = model([im.to(device)])
    pred = pred.append(tempPred)
    
    im = Image.fromarray(im.mul(255).permute(1, 2, 0).byte().numpy())
    sz = pred[0]['boxes'].size()
        # Create figure and axes
    fig, ax = plt.subplots()
        

    ax.imshow(im)        
    color = ["orange","red","green"]
    
    # Tracking:
    if j != 0:
        distances = pred[j][0]['boxes']-pred[j-1][0]['boxes']
    
    for i in range(sz[0]):
        if float(pred[0]['scores'][i]*100) >= 75:
            rect = patches.Rectangle((pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()), pred[j][0]['boxes'][i,2].detach().numpy()-pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,3].detach().numpy()-pred[j][0]['boxes'][i,1].detach().numpy(), linewidth=2, edgecolor=color[pred[j][0]['labels'][i].detach().numpy()], facecolor='none')
            ax.add_patch(rect)
            if float(pred[j][0]['labels'][i].detach().numpy()) == 1:
                ax.annotate('Cola: {0:.2f}%'.format(float(pred[j][0]['scores'][i].detach().numpy()*100)), (pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()-10),color=color[pred[j][0]['labels'][i].detach().numpy()])
            else:
                ax.annotate('Beer: {0:.2f}%'.format(float(pred[j][0]['scores'][i].detach().numpy()*100)), (pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()-10),color=color[pred[j][0]['labels'][i].detach().numpy()])
    plt.show()
if torch.cuda.is_available():
    torch.cuda.empty_cache()