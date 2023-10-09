#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:44:39 2021

- [x] Oriented for local workstation computing 
- [x] VGG16 FE, IN for Dnet
@author: qiwang
"""
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
from utils.pixel_shuffle3d import PixelShuffle3d
from model.model_pretrain_resnet import ResNet, BasicBlock
from model.basic_block import DWT_transform
from model.model_SwinIR import *
def _init_weights(module):
    if isinstance(module, nn.Conv3d):
        module.weight.data.normal_(mean=0., std=1.)
        if module.bias is not None:
            module.bias.data.zero_()

def activation(activation_layer):
    if activation_layer == 'relu':
        return nn.LeakyReLU()
    elif activation_layer == 'GELU':
        return nn.GELU()

def resnet10(**kwargs):
    # default init as kaiming normal already
    model = ResNet(BasicBlock, [1,1,1,1], **kwargs)
    return model

class VGG16(nn.Module):
    def __init__(self,in_channels=1):
        super(VGG16,self) .__init__()
        self.VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(self.VGG16)
        self.apply(_init_weights)

    def create_conv_layers(self,ar):
        layers = []
        in_channels = self.in_channels
        for x in ar:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,stride=1,padding=1),
                       nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2,stride=2)]
                
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv_layers(x)
        return x

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2, activation_layer = 'relu', **kwargs):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.activation = activation(activation_layer)
        self.G_BN = False
        if 'G_BN' in kwargs:
            self.G_BN = kwargs['G_BN']

        def block(in_features, non_linearity=True):
            #layers = [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True, padding_mode='circular')]
            layers = [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                if self.G_BN:
                    layers += [nn.BatchNorm3d(filters),self.activation]
                else:
                    layers += [self.activation]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.blocks = nn.Sequential(*self.blocks)
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self,  filters, res_scale=0.2, **kwargs):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters) )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
class Norm(nn.Module):
    """
    a wrapper module, recording the largest singular values from networks
    """
    def __init__(self,module,name='weight'):
        super(Norm,self).__init__()
        self.module = module
        self.name = name
        self.module.register_buffer(self.name+'_sv', torch.ones(1))
    def update_u_v(self):
        w = getattr(self.module, self.name)
        sv = getattr(self.module, self.name+'_sv')
        sv[:] = torch.norm(w.detach())
        setattr(self.module, self.name+'_sv', sv)
    def forward(self, *args):
        self.update_u_v()
        return self.module.forward(*args)

class Generator(nn.Module): 
    """
    Chnaged to flexible residual blocks
    """
    def __init__(self, channels=1, filters=64, num_res_blocks=3, num_upsample=1):
        super(Generator, self).__init__()

        self.conv1 = Norm(nn.Conv3d(channels, filters, kernel_size=3, stride=1, padding=1))
        # Residual blocks
        self.res_blocks1 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = Norm(nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1))
        # Upsampling layers
        upsample_layers = []
        # activation layer
        self.activation = activation('relu')
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv3d(filters, filters * 8, kernel_size=3, stride=1, padding=1),
                self.activation,
                PixelShuffle3d(scale=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            Norm(nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1)),
            self.activation,
            Norm(nn.Conv3d(filters, channels, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks1(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()

        self.input_shape = (1,64,64,64)
        self.activation = activation(config)
        in_channels, in_height, in_width, in_depth = self.input_shape
        patch_h, patch_w, patch_d = int(in_height / 2 ** 4), int(in_width / 2 ** 4), int(in_depth / 2 ** 4) # meaning of 4: layers of conv
        self.output_shape = (1, patch_h, patch_w, patch_d)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            if config.D_norm_type == 'IN':
                layers.append(nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
                if not first_block:
                    layers.append(nn.InstanceNorm3d(out_filters,affine=True))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
                layers.append(nn.InstanceNorm3d(out_filters,affine=True))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            elif config.D_norm_type == 'BN':
                layers.append(Norm(nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)))
                if not first_block:
                    layers.append(nn.BatchNorm3d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(Norm(nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)))
                layers.append(nn.BatchNorm3d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            elif config.D_norm_type == 'SN':
                layers.append(nn.utils.spectral_norm(nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.utils.spectral_norm(nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)))
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

class Discriminator_SN_SC(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, config, num_in_ch=1, num_feat=64, input_shape = (1,64,64,64),skip_connection=True):
        super(Discriminator_SN_SC, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.input_shape = input_shape
        self.output_shape = input_shape        # the first convolution
        self.conv0 = nn.Conv3d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv3d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv3d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv3d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv3d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv3d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv3d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv3d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

class Discriminator_Unet(nn.Module):
    """ a Unet with SpectraNorm but no skip connection"""
    def __init__(self, config, input_shape:tuple=(1,64,64, 64), num_feature=64,skip_connection=True, n_down_layers:int=3, n_up_layers:int=3):
        super(Discriminator_Unet, self).__init__()
        self.norm = spectral_norm
        self.in_channel = input_shape[0]
        self.input_shape = input_shape
        self.output_shape = (1, int(input_shape[1]/2**2), int(input_shape[2]/2**2),int(input_shape[2]/2**2))
        self.conv_in = nn.Conv3d(self.in_channel, num_feature, 3, 1, 1)
        self.num_features = num_feature
        
        self.layers_down = []
        # down module
        [self.layers_down.extend([self.norm(nn.Conv3d(num_feature*2**i,
            num_feature*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_down_layers)]
        #self.layers_down = self.make_down_layers(n_down_layers)
        self.down = nn.Sequential(*self.layers_down)

        self.layers_up = []
        # up module
        [self.layers_up.extend([self.norm(nn.Conv3d(num_feature*2*2**i,
           num_feature*2**i, 3, 1, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_up_layers-1,-1,-1)]
        #self.layers_up = self.make_up_layers(n_up_layers)
        self.up = nn.Sequential(*self.layers_up)

        self.conv_out = nn.Conv3d(num_feature, 1, 3, 1, 1)
    def forward(self,x):
        x = self.conv_in(x)
        x_low = self.down(x)
        x_high = F.interpolate(x_low, scale_factor=2, mode='trilinear', align_corners = False)
        x_high = self.up(x_high)
        return self.conv_out(x_high)

    def make_up_layers(self, n_layers):
        #layers_up = []
        return [self.layers_up.extend([self.norm(nn.Conv3d(self.num_features*2*2**i,
            self.num_features*2**i, 3, 1, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_layers-1,-1,-1)]

    def make_down_layers(self, n_layers):
        self.layers_down = []
        return [self.layers_down.extend([self.norm(nn.Conv3d(self.num_features*2**i,
            self.num_features*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_layers)]


def blockUNet(in_c, out_c, name, upsample='transpose', bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if upsample=='transpose':
        block.add_module('%s_tconv' % name,nn.Sequential(nn.ConvTranspose3d(in_c, out_c, 4, 2, 1, bias=False),
            DenseResidualBlock(filters=out_c))
            )
    elif upsample == 'downsample':
        block.add_module('%s_conv' % name, nn.Sequential(nn.Conv3d(in_c, out_c, 4, 2, 1, bias=False),
            DenseResidualBlock(filters=out_c))
            ) # 1/2 downsample
    elif upsample == 'linear':
        block.add_module('%s_linear' % name, nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear'),
            DenseResidualBlock(filters=in_c),
            nn.Conv3d(in_c, out_c, 1,1,0, bias=False)
            ))
    # TODO
    #elif upsample=='pixelshuffle':
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm3d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout3d(0.5, inplace=True))
    return block

class dwt_UNet(nn.Module):
    """
    [NOTE]can not apply IWT, cuz both LL and HLs were passed to a conv unit to get channel size not multiplied, this could be explanation of LL band get better approximation, and removal of noise
    """
    def __init__(self,input_shape=(1,64,64,64),output_nc=1, nf=16):
        super(dwt_UNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1, input_shape[1], input_shape[2],input_shape[3])
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv3d(16, nf-1, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8-8, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8-16, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, upsample='downsample', bn=False, relu=False, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 8, nf * 8, name, upsample='transpose', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16+16, nf * 8, name, upsample='transpose', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16+8, nf * 4, name, upsample='transpose', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8+4, nf * 2, name, upsample='transpose', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4+2, nf, name, upsample='transpose', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2+1, nf * 2, name, upsample='linear', bn=True, relu=True, dropout=False)

        self.initial_conv=Norm(nn.Conv3d(1,16,3,padding=1))
        self.bn1=nn.BatchNorm3d(16)
        self.layer1 = layer1 
        self.DWT_down_0= DWT_transform(1,1)
        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)
        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)
        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)
        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv1 = nn.Conv3d(48, 32, 3, padding=1, bias=True)
        self.bn2=nn.BatchNorm3d(32)
        self.tail_conv2 = nn.Conv3d(nf*2, output_nc, 3,padding=1, bias=True)

        self.apply(_init_weights)
    def forward(self, x):
        conv_start=self.initial_conv(x) # (N, 1, H, W, D) -> (N, 16, H, W, D)
        conv_start=self.bn1(conv_start) 
        conv_out1 = self.layer1(conv_start) # (N, 16, H, W, D) -> (N, 15, H/2, W/2, D/2)
        dwt_low_0,dwt_high_0=self.DWT_down_0(x) # (N, 1, H, W, D) -> (N, 1, H/2, W/2, D/2), (N,1,H/2,W/2,D/2)
        out1=torch.cat([conv_out1, dwt_low_0], 1) # -> (N,16,H/2,W/2,D/2)

        conv_out2 = self.layer2(out1) # (N, 16, H/2, W/2, D/2) -> (N, 30, H/4 ,W/4, D/4)
        dwt_low_1,dwt_high_1= self.DWT_down_1(out1) # (N, 16, H/2, W/2, D/2) -> (N, 2, H/4, W/4, D/4), (N, 2, H/4, W/4, D/4)
        out2 = torch.cat([conv_out2, dwt_low_1], 1) # -> (N, 2+30, H/4, W/4, D/4)

        conv_out3 = self.layer3(out2) # (N, 32, H/4, W/4, D/4) -> (N, 60, H/8, W/8, D/8)
        dwt_low_2,dwt_high_2 = self.DWT_down_2(out2) # (N, 32, H/4, W/4, D/4) -> (N, 4, H/8, W/8, D/8), (N, 4, H/8, W/8, D/8)
        out3 = torch.cat([conv_out3, dwt_low_2], 1) # (N, 60+4, H/8, W/8, D/8)

        conv_out4 = self.layer4(out3) # (N, 64, H/8, W/8, D/8) -> (N, 120, H/16, W/16, D/16)
        dwt_low_3,dwt_high_3 = self.DWT_down_3(out3) # (N, 64, H/8, W/8, D/8) -> (N, 8, H/16, W/16, D/16), (N, 8, H/16, W/16, D/16)
        out4 = torch.cat([conv_out4, dwt_low_3], 1) # -> (N, 120+8, H/16, W/16, D/16)

        conv_out5 = self.layer5(out4) # (N, 128, H/16, W/16, D/16) -> (N, 112, H/32, W/32, D/32)
        dwt_low_4,dwt_high_4 = self.DWT_down_4(out4) # (N, 128, H/16, W/16, D/16) -> (N, 16, H/32, w/32, D/32), (N, 16, H/32, W/32, D/32)
        out5 = torch.cat([conv_out5, dwt_low_4], 1) # -> (N, 112+16, H/32, W/32, D/32) 

        out6 = self.layer6(out5) # (N, 128, H/32, W/32, D/32) -> (N, 128, H/64, W/64, D/64)
        dout6 = self.dlayer6(out6) # (N, 128, H/64, W/64, D/64) -> (N, 128, H/32, W/32, D/32)

        Tout6_out5 = torch.cat([dout6, out5, dwt_high_4], 1) # -> (N, 128+128+16, H/32, W/32, D/32)
        Tout5 = self.dlayer5(Tout6_out5) # (N, 272, H/32, W/32, d/32) -> (N, 128, H/16, W/16, D/16)
        Tout5_out4 = torch.cat([Tout5, out4,dwt_high_3], 1)
        Tout4 = self.dlayer4(Tout5_out4)
        Tout4_out3 = torch.cat([Tout4, out3,dwt_high_2], 1)
        Tout3 = self.dlayer3(Tout4_out3)
        Tout3_out2 = torch.cat([Tout3, out2,dwt_high_1], 1)
        Tout2 = self.dlayer2(Tout3_out2)
        Tout2_out1 = torch.cat([Tout2, out1,dwt_high_0], 1)
        Tout1 = self.dlayer1(Tout2_out1)
        Tout1_outinit = torch.cat([Tout1, conv_start], 1)
        tail1=self.tail_conv1(Tout1_outinit)
        tail2=self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)
        return dout1
class DWT_Generator(dwt_UNet):
    def __init__(self):
        super().__init__()
        self.dwt = dwt_UNet()
    def forward(self,x):
        out = F.interpolate(x, scale_factor=2, mode = 'bilinear')
        out1 = self.dwt(out)
        return out1
class dwt_UNet_G(nn.Module):
    def __init__(self,input_shape=(1,64,64,64),output_nc=1, nf=16):
        super(dwt_UNet_G, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (output_nc, input_shape[1], input_shape[2])
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv3d(16, nf-1, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2-2, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4-4, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8-8, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8-16, name, upsample='downsample', bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, upsample='downsample', bn=False, relu=False, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 8, nf * 8, name, upsample='linear', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16+16, nf * 8, name, upsample='linear', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16+8, nf * 4, name, upsample='linear', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8+4, nf * 2, name, upsample='linear', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4+2, nf, name, upsample='linear', bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2+1, nf * 2, name, upsample='linear', bn=True, relu=True, dropout=False)

        self.initial_conv=Norm(nn.Conv3d(1,16,3,padding=1))
        self.bn1=nn.BatchNorm3d(16)
        self.layer1 = layer1
        self.DWT_down_0= DWT_transform(1,1)
        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)
        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)
        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)
        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv1 = nn.Conv3d(48, 32, 3, padding=1, bias=True)
        self.bn2=nn.BatchNorm3d(32)
        self.tail_conv2 = nn.Conv3d(nf*2, output_nc, 3,padding=1, bias=True)

        self.apply(_init_weights)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        conv_start=self.initial_conv(x) # N*in_c*64*64 -> N*16*64*64
        conv_start=self.bn1(conv_start) # -> N*16*64*64 
        conv_out1 = self.layer1(conv_start) # -> N*15*32*32 
        dwt_low_0,dwt_high_0=self.DWT_down_0(x)
        out1=torch.cat([conv_out1, dwt_low_0], 1) # N*16*32*32 ->
        conv_out2 = self.layer2(out1) # -> N*30*16*16 
        dwt_low_1,dwt_high_1= self.DWT_down_1(out1)
        out2 = torch.cat([conv_out2, dwt_low_1], 1) # N*32*16*16 ->
        conv_out3 = self.layer3(out2) # -> N*60*8*8
        dwt_low_2,dwt_high_2 = self.DWT_down_2(out2)
        out3 = torch.cat([conv_out3, dwt_low_2], 1) # N*64*8*8 ->
        conv_out4 = self.layer4(out3) # N*120*4*4
        dwt_low_3,dwt_high_3 = self.DWT_down_3(out3)
        out4 = torch.cat([conv_out4, dwt_low_3], 1) # N*128*4*4 ->
        conv_out5 = self.layer5(out4) # -> N*112*2*2
        dwt_low_4,dwt_high_4 = self.DWT_down_4(out4)
        out5 = torch.cat([conv_out5, dwt_low_4], 1) # N*128*2*2 ->
        out6 = self.layer6(out5) # N*128*1*1 ->
        dout6 = self.dlayer6(out6) # N*128*1*1 -> N*128*2*2

        Tout6_out5 = torch.cat([dout6, out5, dwt_high_4], 1) # N*(128+128+16)*2*2 ->
        Tout5 = self.dlayer5(Tout6_out5) # -> N*128*4*4
        Tout5_out4 = torch.cat([Tout5, out4,dwt_high_3], 1) # N*(128+128+8)*4*4 ->
        Tout4 = self.dlayer4(Tout5_out4) # -> N*64*8*8
        Tout4_out3 = torch.cat([Tout4, out3,dwt_high_2], 1) # N*(64+64+4)*8*8 ->
        Tout3 = self.dlayer3(Tout4_out3) # -> N*32*16*16
        Tout3_out2 = torch.cat([Tout3, out2,dwt_high_1], 1) # N*(32+32+2)*16*16 ->
        Tout2 = self.dlayer2(Tout3_out2) # -> N*16*32*32
        Tout2_out1 = torch.cat([Tout2, out1,dwt_high_0], 1) # N*(16+16+1)*32*32 ->
        Tout1 = self.dlayer1(Tout2_out1) # -> N*32*64*64
        Tout1_outinit = torch.cat([Tout1, conv_start], 1)   # N*(32+16)*64*64 ->
        tail1=self.tail_conv1(Tout1_outinit) # -> N*32*64*64
        tail2=self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)   # -> N*in_c*64*64
        return dout1
#class MWCNN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv_block = blockUNet

class SwinIR(nn.Module):
    def __init__(self, img_size=(32,32,32), patch_size=4, in_chans=1,
                 embed_dim=96, depths=[3, 3, 3, 3], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            #rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_mean = (.5,.5,.5)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # shallow FE
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # deep FE
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1],
                                           patches_resolution[2]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 8, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 8, embed_dim //8, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 8, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1], patches_resolution[2]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w, d = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        mod_pad_d = (self.window_size - d % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W, D = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale, :D*self.upscale]
