#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/torch.out.%j
#SBATCH -e ./log/torch.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=8
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
srun python /u/wangqi/torch_env/crop_gan/test_crop_new.py --path /ptmp/wangqi/MPI_subj3/crops --subj MPRAGE --scale 2
##srun python /u/wangqi/torch_env/crop_gan/crop_GAN.py
echo "Jobs finished"
