import sys, os
sys.path.append('./utils/')
from glob import glob
from multiprocessing import Pool, current_process
from crop_nifti_9t import  crop_3d

"""
A demo of multiprocessing nifti crop, a bit faster than single thread python scrip
"""

def f(filename:str):
    c = crop_3d(filename, step_size=16, save_crops=True)
    print(current_process().name)
    c()

if __name__ == '__main__':
    data_path = sys.argv[1]
    works = list(glob(data_path+'/*.nii*'))
    os.makedirs(data_path+'/crops', exist_ok=True)
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map_async(f, works)
        print(P)
        P.close()
        P.join()
    print("\033[93m Fertig \033[0m")
