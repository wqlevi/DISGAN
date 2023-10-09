from torch.nn.functional import interpolate
import nibabel as nb
from glob import glob
import torch
import sys, os
from multiprocessing import Pool, current_process

# a script using multiprocessing to generate LR whole brain volumes

def make_lr(file_name:str):
    nii = nb.load(file_name)
    img = nii.get_fdata()
    ts = torch.Tensor(img)[None,None,:,:,:]
    lr_ts = interpolate(ts, scale_factor=0.5)
    
    file_name_subj = file_name.split("/")[-1].split(".")[0]
    new_nii = nb.Nifti1Image(lr_ts.squeeze().numpy(), nii.affine)
    nb.save(new_nii, f"{file_name_subj}_lr.nii.gz")

if __name__ == '__main__':
    file_name = sys.argv[1]
    filenames = sorted(glob(file_name+'/*.nii.gz'))
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map_async(make_lr, filenames)
        P.close()
        P.join()
