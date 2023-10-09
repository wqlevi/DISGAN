from swd import swd
import nibabel as nb
import torch
def get_mean_SWD(sr_path:str, gt_path:str, device='cpu', init_slice_idx:int=30):
    # get mean value of Sliced Wasserstein Distance(SWD)
    WD=0
    preslice_idx=init_slice_idx
    sr = torch.from_numpy(nb.load(sr_path).get_fdata())[None,None,:256,:256].float().to(device)
    sr = torch.repeat_interleave(sr, 3, dim=1)
    gt = torch.from_numpy(nb.load(gt_path).get_fdata())[None,None,:256,:256].float().to(device)
    gt = torch.repeat_interleave(gt, 3, dim=1)
    for i in range(preslice_idx, sr.shape[2]-preslice_idx):
        out = swd(sr[:,:,i], gt[:,:,i], device=device)
        WD+=out
    return WD/i

