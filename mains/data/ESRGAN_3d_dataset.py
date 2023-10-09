#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:07:35 2021

- [x] 3d images
- [x] 3d resize and padding
- [x] paired hr,lr images with padding, interpolation
- [x] feature wise normalized
@author: qiwang
"""

import glob
import numpy as np
import nibabel as nb
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

# statistical properties from HCP-Lifespan pilot study (27 yong subjects) dataset
mean = np.array([355.2167])
std = np.array([359.8541])
def denormalize(tensors,channels=1):
    """ Denormalize tensors using mean and std """
    for c in range(channels):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class CropDataset(Dataset):
    def __init__(self,root:str):
        # not normalizing here, since all data is normalized offline
        #self.norm = transforms.Lambda(lambda x : (x - mean)/std)
        self.files = sorted(glob.glob(root + "/*.nii*"))
    def __getitem__(self, index:int):
        img = np.array(nb.load(self.files[index % len(self.files)]).dataobj)
        img = torch.Tensor(img)[None,None,:,:,:]        # n_dim = 5(torch.tensor), for interpolation
        img_hr = img.squeeze(0) # n_dim = 4(torch.tensor)
        img_lr = torch.nn.functional.interpolate(img,32).squeeze(0) # n_dim = 4, tri-linear downsample tensors
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

class Val_Dataset(Dataset):
    def __init__(self,root:str):
        self.norm = transforms.Lambda(lambda x : (x - mean)/std)
        self.files = sorted(glob.glob(root + "/*.nii*"))
    def __getitem__(self, index:int):
        img = nb.load(self.files[index % len(self.files)])
        return torch.Tensor(img.get_fdata())

    def __len__(self):
        return len(self.files)
