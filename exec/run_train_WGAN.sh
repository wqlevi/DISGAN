#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/c18_l1.out.%j
#SBATCH -e ./log/c18_l1.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=1
# wall clock limit(Max. is 24hrs)
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
##srun python /u/wangqi/torch_env/crop_gan/ESRGAN_3d_crop_copy.py --path /ptmp/wangqi/transfer_folder/crops --model C13 --checkpoint 0 
##srun python /u/wangqi/torch_env/crop_gan/train_ESRGAN.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/crops --model Jupyter_c14 --checkpoint 0 --precision 1 --batch_size 6
##srun python /u/wangqi/torch_env/crop_gan/mains/ESRGAN_WGAN_GP.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model_name Jupyter_C18 --checkpoint 26 --precision 1 --batch_size 4 --lr .0002
##srun python /u/wangqi/torch_env/crop_gan/mains/ESRGAN_WGAN_GP_L1.py --train_path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model C18_WGANGP_l1_FE --checkpoint 0 --precision 1 --batch_size 4 --lr .0002
srun python ../mains/ESRGAN_WGAN_GP_L1.py --train_path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model C18_WGANGP_l1_FE --checkpoint 20 --precision 1 --batch_size 4 --lr .0002

echo "Jobs finished"
