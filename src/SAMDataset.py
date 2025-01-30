from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import cv2
from torchvision import transforms
import torch
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cellpose.dynamics import masks_to_flows

class SAMDataset(Dataset):
    def __init__(self, img_path, flow_path, ann_path, processor, weight_path=None, crop_size=256, device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")):
        self.imgs = np.load(img_path)
        self.flows = np.load(flow_path)
        self.anns = np.load(ann_path)
        self.device = device
        if weight_path is not None:
            self.weights = np.load(weight_path)
        else:
            self.weights = None

        self.processor = processor
        self.crop_size = crop_size

        self.isTrain = 'train' in img_path or 'train' in flow_path

    def __len__(self):
        return len(self.imgs)
    
    def _preprocess(self, img):

        #convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #adaptive hist normalization
        # img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)

        #grab 2nd derivative via laplacian
        # edges = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)
        # edges = img_norm

        #grab 1st derivative via sobel
        # edges = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #cvt to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img

    def _data_augmentation(self, img, label, weight, ann):

        ann = ann.astype(np.uint8)

        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            ann = cv2.flip(ann, 1)

            if weight is not None:
                weight = cv2.flip(weight, 1)

        angle_deg = random.randint(-180, 180)
        scale = random.uniform(0.8, 1.2)

        center = (img.shape[1] / 2, img.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, scale)

        # rotate + scale the image
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        ann = cv2.warpAffine(ann, M, (ann.shape[1], ann.shape[0]))

        if weight is not None:
            weight = cv2.warpAffine(weight, M, (weight.shape[1], weight.shape[0]), flags=cv2.INTER_CUBIC)

        # prevent overflow
        img = img.astype(np.float32)
        ann = ann.astype(np.float32)

        # random brightness
        alpha = random.uniform(0.95, 1.05)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        # randomly invert image
        if random.random() > 0.5:
            img = cv2.bitwise_not(img)

        # We must re calculate the flow field after the augmentation as it is dependent on the mask, geometric transformations alter the flows intrinsically

        ann = ann.astype(int)
        # -- 1) Compute flow field (dy, dx) using Cellpose
        # returns array shape (2, H, W)
        flows_2d = masks_to_flows(ann, device=self.device)
        flow_dx = flows_2d[1]
        flow_dy = flows_2d[0]

        # -- 2) Compute a "cell probability" channel using a distance transform
        #    (distance from boundary). We do this on the binary mask of all cells.
        binary_mask = (ann > 0).astype(np.uint8)
        # Distance transform needs a binary image, 0 for background, 1 for foreground
        dist_map = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=3)

        # Normalize to [0,1] to get "cell probability"-like
        max_dist = dist_map.max()
        if max_dist > 0:
            dist_map /= max_dist

        # We call dist_map the "cell_prob" channel
        cell_prob = dist_map.astype(np.float32)

        # Stack them into [3, H, W]
        # channel 0 = dx, channel 1 = dy, channel 2 = cell_prob

        label = np.stack([flow_dx, flow_dy, cell_prob], axis=0)

        return img, label, weight, ann

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image_orig = img.copy()
        label = self.flows[idx]
        ann = self.anns[idx, 0].astype(int)
        if self.weights is not None:
            weight = self.weights[idx]
        else:
            weight = None

        #preprocess image
        img = self._preprocess(img)

        if self.isTrain:
            #data augmentation
            img, label, weight, ann = self._data_augmentation(img, label, weight, ann)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        label = np.stack([label[0][y:y+self.crop_size, x:x+self.crop_size], label[1][y:y+self.crop_size, x:x+self.crop_size], label[2][y:y+self.crop_size, x:x+self.crop_size]], axis=0)
        if self.weights is not None:
            weight = weight[y:y+self.crop_size, x:x+self.crop_size]

        # return image_orig, img, label, weight #just for debugging

        # prepare image and prompt for the model
        inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs['ground_truth_mask'] = label
        # inputs['binary_mask'] = label > 0.001

        if self.weights is not None:
            inputs['weight'] = torch.tensor(weight).float()

        return inputs