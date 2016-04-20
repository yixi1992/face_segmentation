#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=scavenger



python ../../utils/surgery_flow.py \
-f '../modelf1b1/deploy.prototxt' \
-c '/lustre/yixi/face_segmentation_finetune/flow/modelf1b1/snapshots_camvidfmp200200f1b1/train_lr1e-14/_iter_59000.caffemodel' \
-t 'deploy.prototxt' \
-o 'camvid_modelf1b1_surg.caffemodel' \
--fromlayer='conv1_1_f1b1' \
--tolayer='conv1_1_f1b1f2b2'

