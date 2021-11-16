from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import utils
from engine import train_one_epoch, evaluate
import loadData
import transforms as T
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
torch.multiprocessing.freeze_support()

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # Hyper Parameters
    num_classes = 2
    
    
    #torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    # Load Datasets
    data = loadData.Dataset('', None)

    dataset = loadData.Dataset('', get_transform(train=True))
    dataset_test = loadData.Dataset('', get_transform(train=True))

    # Split datasets into train/test
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-100])#:-50
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])#-50:

    #dataset = torch.utils.data.Subset(dataset, indices[:25])#:-50
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-25:])#-50:

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=1,
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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    # move model to the right device
    #train_IMG.to(device)
    #train_Labels.to(device)
    #test_IMG.to(device)
    #test_Labels.to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1,
                                                    gamma=0.01)
    model.to(device)

    # Model training
    num_epochs = 5


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
    # Model Show
    # pick one image from the test set
    img, _ = dataset_test[0]
    
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        
    im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    sz = prediction[0]['boxes'].size()
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(im)
    
    # Create a Rectangle patch
    
    
    # Add the patch to the Axes
    color = ["red","green"]

    for i in range(sz[0]):
        rect = patches.Rectangle((prediction[0]['boxes'][i,0], prediction[0]['boxes'][i,1]), prediction[0]['boxes'][i,2]-prediction[0]['boxes'][i,0], prediction[0]['boxes'][i,3]-prediction[0]['boxes'][i,1], linewidth=2, edgecolor=color[prediction[0]['labels'][i]], facecolor='none')
        ax.add_patch(rect)
    plt.show()

    #Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    #Image.fromarray(prediction[0]['boxes'][0, 0].mul(255).byte().cpu().numpy())
    

    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    sz = prediction[0]['boxes'].size()
    return prediction, model

if __name__ == "__main__":
    test = main()
    model = test[1]
    torch.save(model, 'modelAllpix')
    