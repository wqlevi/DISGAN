import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rot
import sys
import numpy as np
'''
This script crop ROI of brain MRI display in paralle with whole brain image
    Usage:
        * the cropping coordinates are defined as follows:
            --------> x
            |
            |
            |
            V
            y
        * x,y,z slice index are coor in MRICron-1

'''
def plot_save(arr,fname,slice_num=121,coor = (80,120),size=(80,120)):
   
    norm = (lambda x : (x-x.min()) /(x.max()-x.min()))
    fig,axe = plt.subplots(1,1,dpi=300)
    img_whole = np.rot90(arr[slice_num])
    
    axe.imshow(img_whole,cmap='gray',norm=None, vmin=0, vmax=3,interpolation='nearest')
    rect = patches.Rectangle(tuple(map(int,coor)),
                             size[1],size[0],
                             linewidth=3,
                             linestyle='--',
                             edgecolor='yellow',
                             facecolor='none')
    axe.add_patch(rect)
    axe.axis('off')
    plt.savefig(fname[0], transparent=True)
    #-----crop image here----
    fig,axe = plt.subplots(1,1,dpi=300)
    
    img_crop = crop(img_whole,coor, size)
    
    axe.imshow(img_crop,cmap='gray',norm=None,vmin=0,vmax=3,interpolation='nearest')
    axe.axis('off')
    plt.savefig(fname[1], transparent=True, bbox_inches='tight', pad_inches=0)

def crop(arr,coor:tuple, size=(80,120)):
    return arr[int(coor[1]):int(coor[1])+size[0], int(coor[0]):int(coor[0])+size[1]]

def make_fig():
    '''
    argv:
        1 : path to nifti file
        2 : which dataset
    help:
        idx number for x,y are +1 than it was in MRICron, z axis is `SIZE_Z - z_mricron`
    '''
    global data_name
    assert len(sys.argv) >1, "help: python make_fig_new.py <nii path>"
    data_name = sys.argv[1]

    subj = 'LS2009'
    slice_idx = [107,101,143]
    crop_coor=[[80,120],[50,140],[50,68]]
    crop_win_size = (80,120)
    # if not LS2009
    if len(sys.argv) > 2:
        subj = sys.argv[2]
        slice_idx = [130,164,113]
        crop_coor=[[170,70],[70,120],[80,130]]
        crop_win_size = (120,120)
    

    params = {'subject':subj,
             'slice_idx':slice_idx,
             'crop_coor':crop_coor,
             'crop_win_size':crop_win_size
            }
    
    
    arr = rot.load(data_name).get_fdata()#[24:-24,20:-20,10:-10] # resize SR.shape -> HR.shape
    #[:,50:,10:-20] # crop center off background
    global og_coor 
    og_coor = 50,120
    for i,idx, coor_crop in zip(range(3),
                                params['slice_idx'],
                                params['crop_coor']
                                ): # old value: 121,190,96
        arr_view = np.swapaxes(arr, 0, i)

        fname = [data_name.split(".")[0]+f"_{i}_whole.png", data_name.split(".")[0]+f"_{i}_crop.png"]

        plot_save(arr_view,fname, slice_num=idx, coor=coor_crop, size = params['crop_win_size'])
        print(f"Making fig:{fname}")
    return 0
    
if __name__ == '__main__':
    make_fig()

