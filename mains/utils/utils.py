import torch
import torch.nn as nn
import numpy as np
import nibabel as nb
from skimage.metrics import structural_similarity, normalized_root_mse
import matplotlib.pyplot as plt
import yaml

def ssim(image:np.ndarray,gt:np.ndarray):
    '''
    input
    -----
        image: numpy.ndarray
        gt: numpy.ndarray
    output
    -----
        numpy scalar of ssim averaged for channels
    '''
    if not (isinstance(image,np.ndarray) and isinstance(gt,np.ndarray)):
        raise ValueError("both inputs should be in numpy.ndarray type")
    if not image.ndim == gt.ndim:
        raise ValueError("dimensiom of the inputs should be the same")

    data_range = np.max(gt) - np.min(gt)
    if image.ndim==4: # N,H,W,L
        return structural_similarity(image.transpose(1,2,3,0), gt.transpose(1,2,3,0), data_range = data_range, multichannel=True)
    elif image.ndim==3: # H,W,L Batch_size = 1
        return structural_similarity(image, gt, data_range = data_range, channel_axis=-1)

def sharpness(image:np.ndarray):
    '''
    calculating sharpness of a image
    '''
    gradient = np.gradient(image)
    sharpness = np.average(np.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2))
    return sharpness
def psnr(image:np.ndarray,gt:np.ndarray)->np.ndarray:
    mse = np.mean((image - gt)**2)
    if mse == 0:
        return float('inf')
    #    data_range = np.max(gt) - np.min(gt)
    data_range= gt.max() - gt.min() # choose 1 if data in float type, 255 if data in int8
    return 20* np.log10(data_range) - 10*np.log10(mse)

def NRMSE(image, gt):
    '''
    Normalized Root-MSE, by dividing RMSE with max-min
    '''
    return normalized_root_mse(gt,image,normalization='min-max')

def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m,torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    
def initialize_weights(net_l, scale=1):
    # kaiming normal init
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    
def vis_plot(tensor_img):
    # a quick func for visualising the comparisions among tensors
    fig,ax = plt.subplots(3,3,figsize=(10,10))
    axe = ax.ravel()
    img_thickness = int(tensor_img.size(2)/2)
    [axe[i].imshow(tensor_img[i,0,img_thickness].squeeze().detach().cpu().numpy(),cmap='gray') for i in range(tensor_img.size()[0])]
    
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def load_yaml(filename:str):
    yml = yaml.load(open(filename+'.yml'), Loader=yaml.FullLoader)
    return yml

def load_pretrained(model,pretrain_path,replace_key:str='module.',key_dict:str='state_dict'):
    '''
    Parameters
    ----------
        model: model of network
        pretrain_path : str | path to store checkpoint
        key_dict : str | key string of the model(e.g. FE_state_dict, etc.)
    '''
    pretrain_dict = torch.load(pretrain_path)
    net_dict = model.state_dict()
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    return model

def add_noise(arr:np.ndarray, sigma:float=0.)-> np.ndarray:
    assert arr.ndim == 3, "input array should have 3 dimensions"
    return np.random.normal(0., sigma, arr.shape) + arr
