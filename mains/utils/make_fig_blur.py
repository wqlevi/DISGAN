import nibabel as nb
import numpy as np
import sys
import matplotlib.pyplot as plt
from pylab import *
def plot_line(arr):
    rc('axes', linewidth=2)
    font = {
            'family':'serif',
            'color':'black',
            'weight':'normal',
            'size':15,
            }
    if isinstance(arr,list):
        for img, name, color, mkr in zip(arr, ["GT","DCSRN", "Ours", "Linear"], ["#222222","#332FD0","#FB2576","#68B984"], [".","1","*", "x"]):
            plt.plot(np.linspace(0,LEN,LEN),img[Y,X:X+LEN],color,linestyle='--',label=name, marker=mkr,markersize=5)
        plt.tight_layout()
        plt.grid(True, color='#97DECE', linewidth=0.5, linestyle='--')
        plt.xlim(0,LEN)
        plt.legend()
        plt.xlabel("Distance along profile", fontdict=font)
        #plt.ylabel("Voxel intensity", fontdict=font)
        #plt.show()
        plt.savefig(f"line_profile.png",dpi=300,bbox_inches='tight',pad_inches=0)
def plot_fig(arr,outname):
    #plt.plot([X,X+LEN],[Y,Y],"r",linewidth=3)
    plt.imshow(arr,cmap='gray')
    plt.axis('off')
    #plt.title(outname)
    #plt.show()
    plt.savefig(f"results/{outname}.png",dpi=300,bbox_inches='tight',pad_inches=0)

if __name__ == '__main__':
    assert len(sys.argv) >1, "python *.py <input nii path> <output png name>"
    # default order: GT, WGAN, OURS
    nii_list = sys.argv[1:]
    name_list = ["GT","DCSRN","Ours","ArSSR"]
    global X,Y,LEN, X_SLICE, Y_SLICE, Z_SLICE, WIN_SIZE

    #fig.3.b: X,Y,LEN=20,25,20
    X,Y,LEN = 20, 25, 20 # slice for line profile
    # fig.3.a: X=90, Y=101, Z=-180
    # fig.3.b: X=130,Y=180, Z=94
    X_SLICE, Y_SLICE, Z_SLICE = 140,100,-160 # slice for crop: previous X=90,Y=155,Z=-180
    # fig.3.a: WIN_SIZE = 50
    # fig.3.b: WIN_SIZE = 100
    WIN_SIZE = 50
    arr_list = [nb.load(i).get_fdata() for i in nii_list]
    # * change which plane to choose:
    new_arr = [np.rot90(i[X_SLICE:X_SLICE+WIN_SIZE,
                          Y_SLICE,
                          Z_SLICE:Z_SLICE+WIN_SIZE]) for i in arr_list]
    print(new_arr[0].shape)
    #plot_line(new_arr)
    list(plot_fig(new_arr_i,name_i) for new_arr_i, name_i in zip(new_arr,name_list))
