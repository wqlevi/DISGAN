#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/pipe.out.%j
#SBATCH -e ./log/pipe.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:a100:1
# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=8
# #SBATCH --ntasks-per-node=1
# wall clock limit(Max. is 24hrs)
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2020.02
module load cuda/11.2
module load nibabel/2.5.0
# pytorch
module load pytorch/gpu-cuda-11.2/1.8.1

# run
echo "This script aims to crop original -> generate SR -> assemble generated SR"
cd /u/wangqi/git_wq/3d_super-resolution_mri/mains/inference
srun python /u/wangqi/git_wq/3d_super-resolution_mri/mains/inference/inference.py --CkpName instancenoise --epoch 39 --RootPath /ptmp/wangqi/saved_models --hr_path /ptmp/wangqi/transfer_folder/LS200X_Norm/LS2009_demean.nii.gz --save_nii 1

## res10_b16
##srun python /u/wangqi/git_wq/3d_super-resolution_mri/mains/inference/inference.py --CkpName instancenoise_res10_b16 --epoch 3 --model_type 'model_VGG16_IN' --RootPath /ptmp/wangqi/saved_models --hr_path /ptmp/wangqi/transfer_folder/LS200X_Norm/LS2009_demean.nii.gz --save_nii 1
##srun python /u/wangqi/torch_env/crop_gan/crop_GAN.py
echo "Jobs finished"
