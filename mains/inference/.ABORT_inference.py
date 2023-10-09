import nibabel as nb
import numpy as np
import torch
#from numba import njit, range, guvectorize, float32, vectorize
import os
from tqdm import tqdm
import argparse
import sys, glob
sys.path.append("../")

from importlib import import_module
from utils.utils import *
from utils.crop_nifti_9t import crop_3d
from model.model_base import img2lr
import matplotlib.pyplot as plt

def crop_img3d_2list(img_path,crop_size,*step_size,path_save='',norm=False):
    '''
    Args: 
        0: image path for loading
        1: image saving folder
        2: crop_size
        *3: step_size for xyz
    Output: 
        list | coordinate order |
             |    x > y > z     |
        new image shape : tuple
    '''
    print(f'****image\t {img_path.split("/")[-1]} is being cropped****')
    step_size_x,step_size_y,step_size_z = step_size if (len(step_size) == 3) else quit("specify x,y,z step size plz!")
    img = nb.load(img_path)
    
    # pad image
    pad_num = [int((np.ceil(x/64)*64-x)/2) for x in img.get_fdata().shape] # pad all dim to be divided by 64
    img_arr = img.get_fdata()
    if norm:
        img_arr /= img_arr.max()
    pad_img = np.pad(img_arr,((pad_num[0],pad_num[0]),(pad_num[1],pad_num[1]),(pad_num[2],pad_num[2])),'constant',constant_values=(0,))
    pad_nii = nb.Nifti1Image(pad_img,np.eye(4))
    
    imgs = []
    print(pad_img.shape,'\t',img.get_filename())
    for k in range(0,(pad_img.shape[2]-crop_size)+step_size_x,step_size_x):
        for j in range(0,(pad_img.shape[1]-crop_size)+step_size_y,step_size_y):
            for i in range(0,(pad_img.shape[0]-crop_size)+step_size_z,step_size_z):                
                img_c = pad_nii.slicer[i:i+crop_size,j:j+crop_size,k:k+crop_size]
                imgs.append(img_c.get_fdata())
    return imgs,pad_img.shape

            
#@njit(nopython=True, parallel=True)          
def inner_loop(one_arr,crop_ls):
    i = 0 # iterator
    for idz in range(0,Z_NUM):
        for idy in range(0,Y_NUM):
            for idx in range(0,X_NUM):                
                one_arr[CROP_SIZE*idx:CROP_SIZE*(idx+1), CROP_SIZE*idy:CROP_SIZE*(idy+1), CROP_SIZE*idz:CROP_SIZE*(idz+1)] = crop_ls[i]
                i += 1
    return one_arr

def load_model(opt_dict):
    ckp_path = glob.glob(f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/{opt_dict['CkpName']}_Crop_{opt_dict['epoch']}_*.pth")[0]
    ckp = torch.load(ckp_path) # opt_dict['CkpName'] for name of ckp
    model_name = ckp['Gnet_state_dict']
    model_fn = import_module("model."+opt_dict['model_type'])
    print(f"\n=====>Module: {model_fn.__name__} loaded<=====")
    new_model = model_fn.Generator()
    new_model.eval()
    new_model.to(device)
    new_model.load_state_dict(model_name,strict=False)
    return new_model

def assemble_img_X64(crop_list:list,new_shape,scale:int=1):
    global X_NUM,Y_NUM,Z_NUM 
    X_NUM,Y_NUM,Z_NUM = [int(np.ceil(x/64)) for x in new_shape] # num of non-overlapping crops along each dim
    global CROP_SIZE
    CROP_SIZE  = 64*scale
    tmp_arr = np.ones((X_NUM*CROP_SIZE,Y_NUM*CROP_SIZE,Z_NUM*CROP_SIZE),dtype=np.float32) # allocate output image w/ appending size
    out_arr = inner_loop(tmp_arr, crop_list)
    return out_arr  
          
def produce(model,crop_img,scale:int,device,hr_shape=64):
    model.eval()
    with torch.no_grad():
        _,data_tensor_lr = img2lr(crop_img,device,hr_shape,scale)
        res_tensor = model(data_tensor_lr.to(device))
    return res_tensor.detach().squeeze().cpu().numpy()
          

def plot_vis(fake_img,gt,title:str,opt_dict,slice_num=121):
        with torch.no_grad():
            psnr_v = psnr(fake_img,gt)
            ssim_v = ssim(fake_img,gt)
            nrmse_v= NRMSE(fake_img,gt)
        fig = plt.figure(figsize=(48,16))
        ax1 = plt.subplot(131)
        ax1.imshow(np.rot90(fake_img[slice_num]),cmap='gray')
        ax1.title.set_text("2x SR")
        ax2 = plt.subplot(132)
        ax2.imshow(np.rot90(gt[slice_num]),cmap='gray')
        ax2.title.set_text("GT")
        fig.suptitle(title)
        ax3 = plt.subplot(133)
        '''
        _,b,c = ax3.hist(fake_img.ravel(),100,facecolor='g',alpha=0.5,label='fake')
        _,b,c = ax3.hist(gt.ravel(),100,facecolor='b',alpha=0.5,label='GT')
        ax3.set_ylim(0,2000000)
        '''
        ext = gt - fake_img
        im = ax3.imshow(np.rot90(ext[slice_num]),cmap='RdBu_r', vmin = -1, vmax = 1)
        ax_c = ax3.inset_axes([1.04,0.2,0.05,0.6])
        plt.colorbar(im, ax = ax3, cax=ax_c)
        ax1.text(5,220,f"psnr{psnr_v:.3f}",fontdict={'color':'red','size':16,'weight':'bold'})
        ax1.text(5,240,f"SSIM{ssim_v:.3f}",fontdict={'color':'red','size':16,'weight':'bold'})
        ax1.text(5,260,f"NRMSE{nrmse_v:.3f}",fontdict={'color':'red','size':16,'weight':'bold'})
        plt.savefig(f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/"+title+".png")          
if __name__ == "__main__":
    '''
    argv:
        1 : ckp name
        2 : ckp epoch number
        3 : root path dir to ckp
        4 : image full name and path of HR reference
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--CkpName',type=str,help='checkpoint name also saved model folder name')
    parser.add_argument('--epoch',type=int,help='epoch number of the saved ckp')
    parser.add_argument('--RootPath',type=str,help='root path where ckp is stored')
    parser.add_argument('--hr_path',type=str,help='full path to reference HR img')
    parser.add_argument('--save_nii',type=int,default=0,help='full path to reference HR img')
    parser.add_argument('--model_type',type=str,default='model_VGG16_IN',help='module for model to import')
    opt = parser.parse_args()
    opt_dict = opt.__dict__

    # --------global parameters---------- #
    try:
        hr_img_path = opt_dict['hr_path']
    except IndexError:
        hr_img_path = '/big_data/qi1/transfer_folder_HPC/LS2009_demean.nii'
    crop_list,newshape = crop_img3d_2list(f"{hr_img_path}",64,64,64,64)
    device  = torch.device('cuda:0')
    hr_shape = 64 # input orig crop size
    scale = 1
    # --------load model---------- #
    new_model = load_model(opt_dict)
    # --------crop & assemble---------- #
    tmp = [produce(new_model,x,scale,device) for x in crop_list]
    new_img = assemble_img_X64(tmp,newshape,scale)
    if scale == 1:
        tmp_img = new_img[24:-24,10:-10,:] # used for demean_LS2001 scale =1
    elif scale ==2 :
        tmp_img = new_img[48:-48,20:-20,:] # used for demean_LS2001 scale =2
    # --------save & viz---------- #
    hr_nii = nb.load(f"{hr_img_path}")
    hr = hr_nii.get_fdata()
    if opt_dict['save_nii']:
        tmp_nii = nb.Nifti1Image(tmp_img,hr_nii.affine)
        nb.save(tmp_nii,f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/{opt_dict['CkpName']}_epoch_{str(opt_dict['epoch'])}.nii.gz")
    title = opt_dict['CkpName']+"_epoch_"+str(opt_dict['epoch'])
    plot_vis(tmp_img,hr, title, opt_dict)
    plot_vis(tmp_img,hr,title, opt_dict)
    
