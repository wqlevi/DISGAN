import sys, os
sys.path.append('./utils/')
from glob import glob
from multiprocessing import Pool, current_process
from utils import psnr,ssim,NRMSE
from functools import partial
import nibabel as nb

"""
A demo of multiprocessing nifti crop, a bit faster than single thread python scrip
"""

def cal_metrics(sr_name:str,gt_name:str,model_name:str):
    sr,gt = nb.load(sr_name).get_fdata(), nb.load(gt_name).get_fdata()
    with open(f"{model_name}.csv","a") as f:
        print("%f,%f,%f"%(psnr(sr,gt),
            ssim(sr,gt),
            NRMSE(sr,gt)),
            file=f)
if __name__ == '__main__':
    data_path = sys.argv[1]
    gt_path = sys.argv[2]
    works_sr = sorted(glob(data_path+'/*.nii*'))
    works_gt = sorted(glob(gt_path+'/*.nii*'))
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.starmap_async(partial(cal_metrics,model_name=sys.argv[3]), zip(works_sr,works_gt))
        print(P)
        P.close()
        P.join()
    print("\033[93m Fertig \033[0m")
