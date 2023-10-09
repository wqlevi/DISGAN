import numpy as np
from einops import rearrange

def assemble_einops(crop_arr,new_shape:tuple,scale:int=1)->np.ndarray:
    x_num,y_num,z_num = [int(x/64/scale) for x in new_shape]
    tmp_arr = rearrange(crop_arr,'(k1 k2 k3) h w c -> (k3 h) (k2 w) (k1 c)',k1=x_num, k2=y_num, k3=z_num)
    #return tmp_arr.detach().cpu().numpy()
    return tmp_arr
