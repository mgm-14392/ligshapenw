#!/bin/bash

#SBATCH -J biGAN
#SBATCH --partition=c7gpu
#SBATCH --nodelist=brat
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

export CUDA_VISIBLE_DEVICES=0,1,2,3
source activate /c7/home/margonza/miniconda3/envs/NNWS/

python3 eval_byGAN.py -i TRAIN.types \
  -d curated_shapeNW_data \
  -o train/bigan \
  -oplpath train/bigan \
  --GPUIDS 0 1 2 3 \
  --translation 0.0 \
  --shuffle \
  --rotation \
  --batch_size 42 \
  --num_epoch 20000 \
  --nz 8 \
  --lr 0.0005 | tee txtfiles/rerun_biGAN.txt