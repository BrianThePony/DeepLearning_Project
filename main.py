import os
from PIL import Image
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
import loadData
import transforms as T
import random
import math
import sys
import time
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

torch.multiprocessing.freeze_support()

def get_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # Hyper Parameters
    num_classes = 2

    # Transforms
    # Load Datasets
    data = loadData.Dataset('', None)

    dataset = loadData.Dataset('', get_transform(train=True))
    dataset_test = loadData.Dataset('', get_transform(train=True))

    # Split datasets into train/test
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:100])#:-50
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-25:])#-50:

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)



    # Split datasets into train/test
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
    #torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    # move model to the right device
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    lr_scheduler.to(device)

    # %%
    # Model training
    num_epochs = 1


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        print(epoch)
    # Model Show

if __name__ == "__main__":
    main()