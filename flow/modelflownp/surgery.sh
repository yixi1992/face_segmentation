#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=scavenger



python ../../utils/surgery_flow.py \
-f '../modeldefault/deploy.prototxt' \
-c '/lustre/yixi/face_segmentation_finetune/flow/modeldefault/snapshots_tvtcamvidfmp200200/train_lr1e-10/_iter_16000.caffemodel' \
-t 'deploy.prototxt' \
-o 'camvid_modelflownp_tvt_surg.caffemodel' \
--fromlayer='conv1_1' \
--tolayer='conv1_1_flow'

