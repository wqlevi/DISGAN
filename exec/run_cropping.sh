#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/torch.out.%j
#SBATCH -e ./log/torch.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J CROP-CPU

# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="cpu"

# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=8
# #SBATCH --ntasks-per-node=1

# Meme lim
#SBATCH --mem=15000

# wall clock limit(Max. is 24hrs)
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2020.02
module load cuda/11.2
module load nibabel/2.5.0

# run
#srun python /u/wangqi/torch_env/crop_gan/crop_nifti.py /ptmp/wangqi/fb9365 /ptmp/wangqi/fb9365/crops
# srun python /u/wangqi/torch_env/crop_gan/mains/utils/crop_nifti_9t.py /ptmp/wangqi/transfer_folder/LS3017_demean LS 1
srun python /u/wangqi/git_wq/3d_super-resolution_mri/mains/utils/crop_nifti_9t.py /ptmp/wangqi/transfer_folder/anatomy 1
echo "Jobs finished"
