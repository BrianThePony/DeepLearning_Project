import os
import numpy as np
import torch
import torch.utils.data
import cv2
from PIL import Image
import re


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.frames = list(sorted(os.listdir(os.path.join(root, "project_20_data/video1/frames"))))
        self.imgs = list(sorted(os.listdir(os.path.join(root, "project_20_data/video1/imgs"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "project_20_data/video1/imgs", self.imgs[idx])
        frames_path = os.path.join(self.root, "project_20_data/video1/frames", self.frames[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        #mask = np.array(mask)
        # instances are encoded as different colors
        #obj_ids = np.unique(mask)
        # first id is the background, so remove it
        #obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        #num_objs = len(obj_ids)
        text = open(frames_path).read()
        num_objs = len([m.start() for m in re.finditer("<xmin>",text)])
        xminindices = [m.start() for m in re.finditer("<xmin>",text)]
        xminendindices = [m.start() for m in re.finditer("</xmin>",text)]
        xmaxindices = [m.start() for m in re.finditer("<xmax>",text)]
        xmaxendindices = [m.start() for m in re.finditer("</xmax>",text)]
        yminindices = [m.start() for m in re.finditer("<ymin>",text)]
        yminendindices = [m.start() for m in re.finditer("</ymin>",text)]
        ymaxindices = [m.start() for m in re.finditer("<ymax>",text)]
        ymaxendindices = [m.start() for m in re.finditer("</ymax>",text)]
        boxes = []
        for i in range(num_objs):
            xmin = float(text[xminindices[i]+6:xminendindices[i]-1])
            xmax = float(text[xmaxindices[i]+6:xmaxendindices[i]-1])
            ymin = float(text[yminindices[i]+6:yminendindices[i]-1])
            ymax = float(text[ymaxindices[i]+6:ymaxendindices[i]-1])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
