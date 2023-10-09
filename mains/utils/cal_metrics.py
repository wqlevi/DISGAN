import sys
sys.path.append("../")
from utils import psnr, ssim, NRMSE
from glob import glob
import nibabel as nb

# a script for writting main metrics into a cvs file
def eval_metrics(sr_name, gt_name, model_name, noise_level):
    sr,gt = nb.load(sr_name).get_fdata(),nb.load(gt_name).get_fdata()
    with open(f"{model_name}_{noise_level}.csv","a") as f:
        print("%f,%f,%f"%(psnr(sr,gt), ssim(sr,gt), NRMSE(sr,gt)), file=f)


if __name__ == '__main__':
    sr_path = sys.argv[1]
    gt_path = sys.argv[2]
    noise_level = sys.argv[3]
    model_name = sys.argv[4]
    noise_level_str = str(noise_level).replace(".","")
    sr_list = sorted(glob(sr_path+f"/*.nii.gz"))
    gt_list = sorted(glob(gt_path+f"/*.nii.gz"))
    if not sr_path[-1] == '/':
        sr_list = sorted(glob(sr_path+f"*.nii.gz"))
        gt_list = sorted(glob(gt_path+f"*.nii.gz"))

    for sr, gt in zip(sr_list, gt_list):
        eval_metrics(sr,gt,model_name, noise_level)
