#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for cropping 3D volumes into patches, automatically computing residuals and 0-padding.

TODO:
 - [x] From this verion on, all xyz axis will use same step size
 - [x] manually pad axis with 0 to fullfill step_size(constant now = 16)
 - [ ] try to deploy `np.lib.stride_tricks.sliding_window_view(x,(shape_win_x, shape_win_y))[:,::step_size]` to extract patch

USAGE:
    Args:
    1: data path
    2: subject prefix for wildcard search
    3: [optional] flag for padding

@author: qiwang
"""
import nibabel as nb
import numpy as np
import os
import sys
import glob

class crop_3d:
    def __init__(self,data_path,step_size=16,save_nii=False,save_crops=False,**kwargs):
        self.data_path = data_path
        self.step_size = step_size
        self.save_nii = save_nii
        self.save_crops = save_crops
        self.crop_size = 64     
        keys = ['step_size', 'crop_size']
        for k,v in kwargs.items():
            setattr(self, k, v)
    def crop_img3d(self,nii,path_save=None,crop_size=None,step_size=None):
        '''
        Return:
            imgs : list | crops list
            nii.shape : tuple | shape of padded image
        '''
        print(f'****image\t {self.data_path.split("/")[-1]} is being cropped****')
        if crop_size is None:
            crop_size = self.crop_size
        if step_size is None:
            step_size = self.step_size
        if not path_save:
            path_save = self.data_path.rsplit("/",1)[0]
        img = nii
        imgs = []
        print(img.shape,'\t',img.get_filename())
        for k in np.arange(0,(img.shape[2]-crop_size)+step_size,step_size):
            for j in np.arange(0,(img.shape[1]-crop_size)+step_size,step_size):
                for i in np.arange(0,(img.shape[0]-crop_size)+step_size,step_size):
                    img_c = img.slicer[i:i+crop_size,j:j+crop_size,k:k+crop_size]
                    if self.save_crops:
                        img_c = img_c.to_filename(f'{path_save}/crops/{img.get_filename().split("/")[-1].split(".")[0]}_{i}_{j}_{k}.nii') # not using .gz for acceleration
                    else:
                        imgs.append(img_c.get_fdata())
        if not self.save_crops:
            return imgs,nii.shape
        
    # returns residual for padding
    def _check_pad_residual(self, img_arr, crop_size, step_size):
        _shape_exclude_orig = [x-crop_size for x in img_arr.shape]
        return [int((np.ceil((x)/step_size)*step_size-x)/2) for x in _shape_exclude_orig]
    def _check_and_pad_odd(self, img_arr):
        xyz_res = [x%2 for x in img_arr.shape] # residual for even shape
        img_arr = img_arr[:img_arr.shape[0]-xyz_res[0],
                :img_arr.shape[1]-xyz_res[1],
                :img_arr.shape[2]-xyz_res[2]] # reduce 1 entry if odd shape
        return img_arr

    def pad_new_nii(self,nii):
        img = nii
        img_arr = img.get_fdata()
        img_arr = self._check_and_pad_odd(img_arr)
        pad_num = self._check_pad_residual(img_arr, self.crop_size, self.step_size)
        pad_tuple = tuple((i,i) for i in pad_num)
        print(pad_tuple)
        pad_nii = np.pad(img_arr,pad_tuple,'constant',constant_values=(0,))
        if self.save_nii:
            pad_nii = pad_nii.astype('<i2')
            img_save = nb.Nifti1Image(pad_nii, img.affine, img.header)
            nb.save(img_save,img.get_filename())
        else:
            pad_nii = pad_nii.astype('<f8') # change to <i2 for int16 in 3T | <f8 for float64 in 9T
            img_nii = nb.Nifti1Image(pad_nii, img.affine)
            img_nii.set_filename(nii.get_filename())
            return img_nii
    def __call__(self):
        x = nb.load(self.data_path)
        x = self.pad_new_nii(x)
        return self.crop_img3d(x)
    
if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = data_path+'/crops'
    os.makedirs(save_path,exist_ok = True)
    for fname in glob.glob(data_path+'/*nii*'):
        c = crop_3d(fname, step_size=16,save_crops=True)
        c()
