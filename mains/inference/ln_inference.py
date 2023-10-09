import nibabel as nb
import numpy as np
import torch
#from numba import njit, range, guvectorize, float32, vectorize
import os
import argparse
import sys, glob
sys.path.append("../")

from ln_DDP_train import GAN
from importlib import import_module
from utils.utils import *
from utils.crop_nifti_9t import crop_3d
from model.model_base import img2lr 
import matplotlib.pyplot as plt
import matplotlib as mpl
from inference import load_model

"""
This version of inference allows overlapped stiching(modify "step_size" in function "assemble")
"""
# Asseble with overlap
#profile
def assemble(crop_list, new_shape, **kwargs):
    counter, crop_size, step_size=0, 64, 64
    if 'crop_size' in kwargs.keys():
        crop_size = kwargs['crop_size']
    if 'step_size' in kwargs.keys():
        step_size = kwargs['step_size']
    img_empty = np.empty(new_shape)
    for k in range(0,(new_shape[2]-crop_size)+step_size,step_size):
        for j in range(0,(new_shape[1]-crop_size)+step_size,step_size):
            for i in range(0,(new_shape[0]-crop_size)+step_size,step_size):   
                img_empty[i:i+crop_size,j:j+crop_size,k:k+crop_size] = crop_list[counter]
                counter+=1
    return img_empty


def ln_load_G(opt_dict):
    ckp_path = glob.glob(f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/{opt_dict['CkpName']}_Crop_{opt_dict['epoch']}*.ckpt")[0]
    model = GAN.load_from_checkpoint(ckp_path)
    Gnet = model.generator
    Gnet.eval()
    Gnet.to(device)
    return Gnet


          
#profile
def produce(model,crop_img,scale:int,device,hr_shape=64):
    # takes 17sec, largest overhead
    model.eval()
    with torch.no_grad():
        _,data_tensor_lr = img2lr(crop_img,device,hr_shape,scale)
        res_tensor = model(data_tensor_lr)
    #return res_tensor.detach().squeeze().cpu().numpy() # detach is the overhead, around 15sec
    return res_tensor.squeeze() # detach is the overhead, around 15sec
          
#profile
def center_crop(img,cropx,cropy,cropz):
    x,y,z = img.shape
    start_x = x//2-(cropx//2)
    start_y = y//2-(cropy//2)
    start_z = z//2-(cropz//2)
    return img[start_x:start_x+cropx, start_y:start_y+cropy, start_z:start_z+cropz]

#profile
def plot_vis(fake_img,gt,title:str,opt_dict,slice_num=121):
        with torch.no_grad():
            psnr_v = psnr(fake_img,gt)
            ssim_v = ssim(fake_img,gt)
            nrmse_v= NRMSE(fake_img,gt)
        if not opt_dict['log_metrics']:
            print(f"psnr{psnr_v:.3f}\n")
            print(f"ssim{ssim_v:.3f}\n")
            print(f"nrmse{nrmse_v:.3f}\n")
        else:
            with open(f"{opt_dict['CkpName']}.txt","a+") as f:
                f.write(str(psnr_v)+","+str(ssim_v)+","+str(nrmse_v)+"\n")
                f.close()
        #plt.savefig(f"{opt_dict['RootPath']}/{opt_dict['CkpName']}/"+title+".png",transparent=True)         
#profile
def main(opt_dict):
    # --------global parameters---------- #
    
    global device
    device  = torch.device('cuda:0')
    hr_shape = 64 # input orig crop size

    hr_img_path = opt_dict['hr_path']
    Crop = crop_3d(f"{hr_img_path}",step_size = opt_dict['step_size'], crop_size=hr_shape)
    crop_list,newshape = Crop()
    scale = 1

    # --------load model---------- #
    Gnet = load_model(opt_dict, use_ln=True)
    #Gnet = ln_load_G(opt_dict)

    # --------crop & assemble---------- #
    tmp = torch.stack([produce(Gnet,x,scale,device, hr_shape=hr_shape) for x in crop_list]).detach().cpu()
    new_img = assemble(tmp,newshape,
            step_size = opt_dict['step_size'],
            crop_size = hr_shape)
    
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
    plot_vis(new_img,hr,title, opt_dict)
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
    parser.add_argument('--step_size',type=int,default=16,help='epoch number of the saved ckp')
    parser.add_argument('--RootPath',type=str,help='root path where ckp is stored')
    parser.add_argument('--hr_path',type=str,help='full path to reference HR img')
    parser.add_argument('--ckp_path',type=str,help='full path to reference HR img')
    parser.add_argument('--save_nii',type=int,default=0,help='full path to reference HR img')
    parser.add_argument('--model_type',type=str,default='model_VGG16_IN',help='module for model to import')
    parser.add_argument('--log_metrics',type=int,default=0,help='decide whether to write metrics to file')
    try:
        opt = parser.parse_args()
    except:
        parser.print_help()
        sys.exit("NO args")
    opt_dict = opt.__dict__
    print(opt.__dict__)

    main(opt_dict) 


    opt_dict = {'CkpName':'instancenoise_res10_b16','epoch':15,'RootPath':'/big_data/qi1/saved_models','hr_path':'/big_data/qi1/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001/BraTS20_Validation_001_t1_demean.nii.gz','save_nii':1,'model_type':'model_VGG16_IN'}
