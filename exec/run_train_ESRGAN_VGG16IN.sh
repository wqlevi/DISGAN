#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/inst_n.out.%j
#SBATCH -e ./log/inst_n.err.%j
# initial working dir:
#SBATCH -D /u/wangqi/torch_env/crop_gan
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

## resnet10 batchsize 16 two subjects as trains sample
srun python /u/wangqi/git_wq/test_3d_SR/mains/train_script_instnoise_wandb.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/LS_all_subj_norm/crops --path_save /ptmp/wangqi/saved_models --model instancenoise_res10_b16_1subj --checkpoint 4 --precision 1 --batch_size 16 --epochs 6 --lr .0001 --inst_noise 0 --FE_type 'resnet10' --D_type 'Discriminator_Unet' --update_FE

echo "Jobs finished"
