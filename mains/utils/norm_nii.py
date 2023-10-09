import nibabel as nb
import numpy as np
from glob import glob
import sys,os

def norm_inner_loop(filename:str, parent_dir:str, save_path:str):
    nii = nb.load(parent_dir+"/"+filename)
    arr = nii.get_fdata()
    arr = (arr - arr.mean())/arr.std()
    new_nii = nb.Nifti1Image(arr, nii.affine)
    nb.save(new_nii,save_path+"/"+filename.split(".")[0]+"_demean.nii.gz")

def norm_outer_loop(rootpath:str, savepath:str):
    if rootpath.split(".")[-1] == "nii" or rootpath.split(".")[-1] == "gz": # when rootpath is one image
        filename = rootpath.rsplit("/",1)[-1]                               # filename with suffix
        parent_dir = rootpath.rsplit("/",1)[0]                              # parent dir of image
        norm_inner_loop(filename, parent_dir, savepath)
    else:                                                                   # when rootpath is a dir of images
        filename = [f.split("/")[-1] for f in glob(rootpath+"*.nii*")]
        if len(filename) == 0:
            raise ValueError 
        parent_dir = rootpath
        [norm_inner_loop(name, parent_dir, savepath) for name in filename]
    
    

if __name__ == '__main__':
    #    norm = lambda(x: (x-x.mean())/x.std())
    root_path = sys.argv[1] # path for data root, end with /
    new_path =  sys.argv[2] # path for new crops, end without / 
    os.makedirs(new_path, exist_ok=True)
    norm_outer_loop(root_path, new_path)
        
