#module load anaconda/3/2020.02
#module load cuda/11.2
#module load nibabel/2.5.0
#module load pytorch/gpu-cuda-11.2/1.8.1

python test_3d_crop_ESRGAN.py --idx 16 --path ../utils/LS2001 --subj LS2001

