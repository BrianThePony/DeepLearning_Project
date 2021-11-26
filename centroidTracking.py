# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:11:51 2021

@author: magnu
"""

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np



class centroidTracker():
    def __init__(self,maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.object[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        
        inputCentroids = np.zeros((len(rects),2),dtype="int")
        
        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            
            cX = int((startX + endX) / 2)
            cY = int((startY + endY) / 2)
            inputCentroids[i] = (cX, cY)
            
            if len(self.objects) == 0:
                for i in range(0, len(inputCentroids)):
                    self.register(inputCentroids[i])
                    
            else:
                
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())
                
                
                
                D = dist.cdist(np.array(objectCentroids),inputCentroids)
                
                rows = D.min(axis=1).argsort()
                
                cols = D.argmin(axis=1)[rows]
                
                usedRows = set()
                
                usedCols = set()
                
                for (row,col) in zip(rows,cols):
                    
                    if row in usedRows or col in usedCols:
                        continue
                    
                    objectID = objectIDs[row]
                    
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    
                    usedRows.add(row)
                    usedCols.add(col)
                    
                unusedRows = set(range(0,D.shape[0])).difference(usedRows)
                    
                unusedCols = set(range(0,D.shape[1])).difference(usedCols)
                
                if D.shape[0] >= D.shape[1]:
                    
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID] += 1
                        
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
                            
                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col])
        return self.objects
#%%
from scipy.spatial import distance as dist
import numpy as np

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime
import imageio

ct = centroidTracker()

(H, W) = (None, None)

print("[INFO] loading model...") 
            
model_path = r"modelAllpixtest2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path,map_location=device)

print("[INFO] starting video stream...")
use_test_video = True
if use_test_video:
    root = r".\project_20_data\video1.avi"
    vs = cv2.VideoCapture(root)
    idx = 0
else:
    vs = VideoStream(src = 0).start()
time.sleep(2.0)

while True:
    if use_test_video:
        idx += 5
        vs.set(cv2.CAP_PROP_POS_FRAMES,1681+idx);
    frame = vs.read()
    frame = frame[1]
    frame = imutils.resize(frame, width = 320)
    frame = np.transpose(frame,(2,0,1)) # Convert image to correspond to expected model input
    
    model.eval()
    frame_np = torch.from_numpy(frame[[2,0,1],:,:]) # Change RGB order
    frame_np = frame_np/255
    with torch.no_grad():
        tempPred = model([frame_np.to(device)])
    
    sz = tempPred[0]['boxes'].size()
    
    score_thresh = 0.90
    
    frame = np.transpose(frame,(1,2,0)) # Convert image back to before model
    
    rects = []
    for i in range(sz[0]):
        if tempPred[0]['scores'].cpu()[i].item() > score_thresh:
            box = tempPred[0]['boxes'][i].cpu()
            rects.append(np.array([box[0].item(), box[1].item(), box[2].item(), box[3].item()]).astype("int"))
            if tempPred[0]['labels'].cpu()[i].item() == 1:
                cv2.rectangle(frame, (int(box[0].item()), int(box[1].item())) , (int(box[2].item()), int(box[3].item())),(0, 0, 255), 2) 
            else:
                cv2.rectangle(frame, (int(box[0].item()), int(box[1].item())) , (int(box[2].item()), int(box[3].item())),(0, 255, 0), 2)
        
    objects = ct.update(rects)
    
    
    for (objectID, centroid) in objects.items():
        
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        cv2.circle(frame, (centroid[0], centroid[1]),4, (0,255,0),2)
                    
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  
    
    
    
    if key == ord("q"):
        break
    

print("Program stopped")
cv2.destroyAllWindows()
# %%

