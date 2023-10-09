import nibabel as nb
import numpy as np
from glob import glob
import sys,os
from multiprocessing import Pool, current_process
from functools import partial

"""
multiprocessing for pre-processing, 0-mean 1-std in this case
"""
def norm_inner_loop(filename:str):
    nii = nb.load(filename)
    arr = nii.get_fdata()
    arr = (arr - arr.mean())/arr.std()
    new_nii = nb.Nifti1Image(arr, nii.affine)
    save_filename = filename.split(".",1)[0]
    print(save_filename)
    nb.save(new_nii,save_filename+"_demean.nii.gz")

def spawn(root_path):
    filenames = sorted(glob(root_path+"/*.nii*"))
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map_async(norm_inner_loop, filenames)
        P.close()
        P.join()
    

if __name__ == '__main__':
    root_path = sys.argv[1] # path for data root, end with /
    spawn(root_path)
        
