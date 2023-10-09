import nibabel as nb
import numpy as np
import torch
#from numba import njit, range, guvectorize, float32, vectorize
import os
import argparse
import sys, glob
sys.path.append("../")

from importlib import import_module
from utils.utils import *
from utils.crop_nifti_9t import crop_3d
from model.model_base import img2lr, load_pretrained
import matplotlib.pyplot as plt
import matplotlib as mpl


def inner_loop(one_arr,crop_ls):
    i = 0 # iterator
    for idz in range(0,Z_NUM):
        for idy in range(0,Y_NUM):
            for idx in range(0,X_NUM):                
                one_arr[CROP_SIZE*idx:CROP_SIZE*(idx+1), CROP_SIZE*idy:CROP_SIZE*(idy+1), CROP_SIZE*idz:CROP_SIZE*(idz+1)] = crop_ls[i]
                i += 1
    return one_arr

def load_model(opt_dict,*,state_key = 'state_dict', **kwargs):
    """
    opt_dict: [dict] 'ckp_path', 'model_type'
    kwargs: 'use_ln [BOOL]'
    """
        
    ckp_path = opt_dict['ckp_path']

    device = "cuda:0"
    if 'device' in kwargs:
        device = kwargs['device']
    model_fn = import_module("model."+opt_dict['model_type'])
    print(f"\n=====>Module: {model_fn.__name__} loaded<=====")
    new_model = getattr(model_fn, opt_dict['G_type'])()
    new_model.eval()
    new_model.to(device)
    new_model = load_pretrained(new_model,ckp_path,state_key,**kwargs)
    return new_model

def assemble_img_X64(crop_list,new_shape,scale=1):
    global X_NUM,Y_NUM,Z_NUM 
    X_NUM,Y_NUM,Z_NUM = [int(np.ceil(x/64)) for x in new_shape] # num of non-overlapping crops along each dim
    global CROP_SIZE
    CROP_SIZE  = 64*scale
    tmp_arr = np.ones((new_shape),dtype=np.float32) # allocate output image w/ appending size
    out_arr = inner_loop(tmp_arr, crop_list)
    return out_arr  
          
def produce(model,crop_img,scale:int,device,hr_shape=64):
    model.eval()
    with torch.no_grad():
        _,data_tensor_lr = img2lr(crop_img,device,hr_shape,scale)
        res_tensor = model(data_tensor_lr)
    return res_tensor.squeeze()           

def center_crop(img,cropx,cropy,cropz):
    x,y,z = img.shape
    start_x = x//2-(cropx//2)
    start_y = y//2-(cropy//2)
    start_z = z//2-(cropz//2)
    return img[start_x:start_x+cropx, start_y:start_y+cropy, start_z:start_z+cropz]

def main(opt_dict):
    # --------global parameters---------- #
    try:
        hr_img_path = opt_dict['hr_path']
    except IndexError:
        hr_img_path = '/big_data/qi1/transfer_folder_HPC/LS2009_demean.nii'
    Crop = crop_3d(f"{hr_img_path}",step_size = 64)
    crop_list,newshape = Crop()
    global device
    device  = torch.device('cuda:0')
    hr_shape = 64 # input orig crop size
    scale = 1

    # --------load model---------- #
    new_model = load_model(opt_dict)

    # --------crop & assemble---------- #
    tmp = torch.stack([produce(new_model,x,scale,device) for x in crop_list]).detach().cpu()
    new_img = assemble_img_X64(tmp,newshape,scale)
    
    # --------save & viz---------- #
    hr_nii = nb.load(f"{hr_img_path}")
    new_img = center_crop(new_img,*hr_nii.shape)
    #hr_nii = Crop.pad_new_nii(hr_nii)
    hr = hr_nii.get_fdata()
    if opt_dict['save_nii']:
        os.makedirs(f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/saved_nii", exist_ok = True)
        tmp_nii = nb.Nifti1Image(new_img,hr_nii.affine)
        nb.save(tmp_nii,f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/saved_nii/{opt_dict['CkpName']}_{opt_dict['hr_path'].split('/')[-1]}_epoch_{str(opt_dict['epoch'])}.nii.gz")
    title = opt_dict['CkpName']+"_epoch_"+str(opt_dict['epoch'])
#%%
if __name__ == "__main__":
    """
    argv:
        1 : ckp name
        2 : ckp epoch number
        3 : root path dir to ckp
        4 : image full name and path of HR reference
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--CkpName',type=str,help='checkpoint name also saved model folder name')
    parser.add_argument('--epoch',type=int,help='epoch number of the saved ckp')
    parser.add_argument('--RootPath',type=str,help='root path where ckp is stored')
    parser.add_argument('--ckp_path',type=str,help='root path where ckp is stored')
    parser.add_argument('--hr_path',type=str,help='full path to reference HR img')
    parser.add_argument('--save_nii',type=int,default=0,help='full path to reference HR img')
    parser.add_argument('--model_type',type=str,default='model_VGG16_IN',help='module for model to import')
    parser.add_argument('--G_type',type=str,default='Generator',help='module for model to import')
    parser.add_argument('--log_metrics',type=int,default=0,help='decide whether to write metrics to file')
    opt = parser.parse_args()
    opt_dict = opt.__dict__
    print(opt.__dict__)

    main(opt_dict) 


