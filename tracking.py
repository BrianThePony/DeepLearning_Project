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
from datetime import datetime
import imageio
imlist = []
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
start = datetime.now()
root = r".\tracking"
for i in list((os.listdir(root))):
    im = Image.open(os.path.join(root,i)).convert('RGB')
    im = np.asarray(im)
    im = np.transpose(im,(2,0,1))#np.transpose(im)
    imlist.append(im)
#imlist = np.asarray(imlist)
#imlist = torch.FloatTensor(imlist)
model_path = r".\modelAllpixtest2"
pred = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filenames = []
for j in range(len(imlist)):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = torch.load(model_path,map_location=device)
    model.eval()
    im = torch.from_numpy(imlist[j])
    im = im/255
    with torch.no_grad():
        tempPred = model([im.to(device)])
    pred.append(tempPred)
    
    im = Image.fromarray(im.mul(255).permute(1, 2, 0).byte().numpy())
    sz = pred[j][0]['boxes'].size()
        # Create figure and axes
    fig, ax = plt.subplots()
        
    score_thresh = 80 # At what percentage score does the box count as a can
    
    ax.imshow(im)        
    color = ["orange","red","green"]
    
    # Tracking:
        
    if torch.cuda.is_available():
        if j == 0:
            ID = np.linspace(1,sum(pred[j][0]['scores'].cpu()*100>=score_thresh),sum(pred[j][0]['scores'].cpu()*100>=score_thresh),dtype=int)
        elif j != 0:
            IDold = ID
            ID = np.linspace(1,sum(pred[j][0]['scores'].cpu()*100>=score_thresh),sum(pred[j][0]['scores'].cpu()*100>=score_thresh),dtype=int)
    
            min_distance = np.zeros(np.size(ID))
            nearest_neighbors = np.zeros(np.size(ID),dtype=int)
            for k in range(sum(pred[j][0]['scores'].cpu()*100 >= score_thresh)):
                distances = []
                for l in pred[j-1][0]['boxes'][pred[j-1][0]['scores'].cpu()*100>=score_thresh,:]:
                    distances.append(torch.linalg.norm(pred[j][0]['boxes'][k,:].cpu()-l.cpu()))
                min_distance[k] = min(distances)
                nearest_neighbors[k] = distances.index(min_distance[k])
            if len(ID) > len(IDold) or len(ID) < len(IDold):
                min_min_distance = min(min_distance)
                ID[int(nearest_neighbors[min_distance.tolist().index(min_min_distance)])] = IDold[int(nearest_neighbors[min_distance.tolist().index(min_min_distance)])]
            else:
                ID = IDold[nearest_neighbors]
    else:
        if j == 0:
            ID = np.linspace(1,sum(pred[j][0]['scores']*100>=score_thresh),sum(pred[j][0]['scores']*100>=score_thresh),dtype=int)
        elif j != 0:
            IDold = ID
            ID = np.linspace(1,sum(pred[j][0]['scores']*100>=score_thresh),sum(pred[j][0]['scores']*100>=score_thresh),dtype=int)
    
            min_distance = np.zeros(np.size(ID))
            nearest_neighbors = np.zeros(np.size(ID),dtype=int)
            for k in range(sum(pred[j][0]['scores']*100 >= score_thresh)):
                distances = []
                for l in pred[j-1][0]['boxes'][pred[j-1][0]['scores']*100>=score_thresh,:]:
                    distances.append(torch.linalg.norm(pred[j][0]['boxes'][k,:]-l))
                min_distance[k] = min(distances)
                nearest_neighbors[k] = distances.index(min_distance[k])
            if len(ID) > len(IDold) or len(ID) < len(IDold):
                min_min_distance = min(min_distance)
                ID[int(nearest_neighbors[min_distance.tolist().index(min_min_distance)])] = IDold[int(nearest_neighbors[min_distance.tolist().index(min_min_distance)])]
            else:
                ID = IDold[nearest_neighbors]
            
    
    for i in range(sz[0]):
        if float(pred[j][0]['scores'][i]*100) >= score_thresh:
            if torch.cuda.is_available():
                rect = patches.Rectangle((pred[j][0]['boxes'][i,0], pred[j][0]['boxes'][i,1]), pred[j][0]['boxes'][i,2]-pred[j][0]['boxes'][i,0], pred[j][0]['boxes'][i,3]-pred[j][0]['boxes'][i,1], linewidth=2, edgecolor=color[pred[j][0]['labels'][i]], facecolor='none')
                ax.add_patch(rect)
                if float(pred[j][0]['labels'][i]) == 1:
                    ax.annotate('Cola {0:d}: {1:.2f}%'.format(ID[i],float(pred[j][0]['scores'][i]*100)), (pred[j][0]['boxes'][i,0], pred[j][0]['boxes'][i,1]-10),color=color[pred[j][0]['labels'][i]])
                else:
                    ax.annotate('Beer {0:d}: {1:.2f}%'.format(ID[i],float(pred[j][0]['scores'][i]*100)), (pred[j][0]['boxes'][i,0], pred[j][0]['boxes'][i,1]-10),color=color[pred[j][0]['labels'][i]])
            else:
                rect = patches.Rectangle((pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()), pred[j][0]['boxes'][i,2].detach().numpy()-pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,3].detach().numpy()-pred[j][0]['boxes'][i,1].detach().numpy(), linewidth=2, edgecolor=color[pred[j][0]['labels'][i].detach().numpy()], facecolor='none')
                ax.add_patch(rect)
                if float(pred[j][0]['labels'][i].detach().numpy()) == 1:
                    ax.annotate('Cola {0:d}: {1:.2f}%'.format(ID[i],float(pred[j][0]['scores'][i].detach().numpy()*100)), (pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()-10),color=color[pred[j][0]['labels'][i].detach().numpy()])
                else:
                    ax.annotate('Beer {0:d}: {1:.2f}%'.format(ID[i],float(pred[j][0]['scores'][i].detach().numpy()*100)), (pred[j][0]['boxes'][i,0].detach().numpy(), pred[j][0]['boxes'][i,1].detach().numpy()-10),color=color[pred[j][0]['labels'][i].detach().numpy()])
    filename = f'gifMaker/{j}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# build gif
with imageio.get_writer('test.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)
    
print(datetime.now() - start)