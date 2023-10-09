#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/assem.out.%j
#SBATCH -e ./log/assem.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --constraint="cpu"
# Number of nodes and MPI tasks per node:
#SBATCH --mem=64000
# #SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
# wall clock limit(Max. is 24hrs)
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2020.02
module load nibabel/2.5.0
# pytorch
##module load pytorch/gpu-cuda-11.2/1.8.1

# run
srun python /u/wangqi/torch_env/crop_gan/assemble_crop_v3.py --path /ptmp/wangqi/MPI_subj3/gen_data --subj MPRAGE --scale 2
echo "Jobs finished"
