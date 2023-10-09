#!/bin/bash -l
### Paths for use ###
HOME_DIR='/u/wangqi'
SRC_DIR='/u/wangqi/git_wq/3d_super-resolution_mri/mains'
DATA_DIR='/ptmp/wangqi/LS_all/crops'

# Standard output and error:
#SBATCH -o $HOME_DIR/log/mri_sr.out.%j
#SBATCH -e $HOME_DIR/log/mri_sr.err.%j
# initial working dir:
#SBATCH -D $SRC_DIR
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:rtx5000:2
# Number of nodes and MPI tasks per node:
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=1
# wall clock limit(Max. is 24hrs)
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2021.11
module load gcc/11
module load openmpi/4
# pytorch
module load pytorch-distributed/gpu-cuda-11.6/2.0.0
module load pytorch-lightning/2.0.1

srun python ln_DDP_train.py --model_name 'DWT_D'
echo "Jobs finished"
