#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:44:39 2021

- [x] all on same GPU A100, Raven. Remote visualization session
- [x] same as ESRGAN but no BN in Dnet  
@author: qiwang
"""
import torch.nn as nn
import torch
from utils import PixelShuffle3d
from model.model_DISGAN import Norm, VGG16, DenseResidualBlock

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        # personalized BN layer for Gnet, added to WGAN-GP for stablizing gradient
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters), nn.BatchNorm3d(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class Discriminator(nn.Module):
    '''
    This Discriminator is the same as critic network, which has no last layer of nonlinear activation
    Both active in WGAN-GP and Relativistic GAN
    Config:
    ------
        BN turned off for WGAN-GP
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_shape = (1,64,64,64) # hr image shape
        in_channels, in_height, in_width, in_depth = self.input_shape
        patch_h, patch_w, patch_d = int(in_height / 2 ** 4), int(in_width / 2 ** 4), int(in_depth / 2 ** 4) # meaning of 4: layers of conv
        self.output_shape = (1, patch_h, patch_w, patch_d) # Dnet output shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(Norm(nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)))
            
            if not first_block: # used to be commented for no BN on Dnet
                layers.append(nn.BatchNorm3d(out_filters))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(Norm(nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)))#half size
            layers.append(nn.BatchNorm3d(out_filters)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(Norm(nn.Conv3d(out_filters, 1, kernel_size=3, stride=1, padding=1))) # 4 Conv layers in block

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
