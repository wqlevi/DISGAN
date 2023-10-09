#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/c20.out.%j
#SBATCH -e ./log/c20.err.%j
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

## FE without loading pretrianed weight:
##srun python /u/wangqi/torch_env/crop_gan/mains/train_ESRGAN_res10.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model pre-trained_c15 --checkpoint 38 --precision 1 --batch_size 4 --lr .0002

## FE with pretrained weight resnet10.pth:
##srun python /u/wangqi/torch_env/crop_gan/mains/train_ESRGAN_res10.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model pre-trained_c16 --checkpoint 0 --precision 1 --batch_size 4 --lr .0002

## FE with pretrained weigth resnet50.pth: [FAILED]

## FE with updating ResNet-10 1G1D:
##srun python /u/wangqi/torch_env/crop_gan/mains/train_ESRGAN_res10.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model pre-trained_c17 --checkpoint 19 --precision 1 --batch_size 4 --lr .0002 --update_ratio 1 --update_FE 1

## normal resnet10_FE experiment:
## srun python /u/wangqi/git_wq/3d_super-resolution_mri/mains/train_script_resnet10_tsboard.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model pre-trained_c19 --checkpoint 0 --precision 1 --batch_size 8 --lr .0002  --update_FE 0

## Pixel Attention Network experiment:(4Juni2022)
##srun python /u/wangqi/torch_env/crop_gan/mains/train_script_PAN_tsboard_FE.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model C20_FE --checkpoint 0 --precision 1 --batch_size 8 --lr .0002  --pretrained_FE 0 --update_FE 0 --epoch 100

##8Juni2022
srun python /u/wangqi/torch_env/crop_gan/mains/train_script_PAN_tsboard_FE.py --path /ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops --model C20_FE_VGG --checkpoint 20 --precision 1 --batch_size 4 --lr .0002  --pretrained_FE 0 --update_FE 0 --epoch 20 --type_FE VGG16 


echo "Jobs finished"
