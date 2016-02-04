#!/bin/bash

#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --gres=gpu

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh

python ./solve.py > solve.out
#./caffe/build/tools/caffe train -solver face_segmentation_finetune_solver.prototxt -weights ../image+jit2d+illum+rot3d/snapshots/snapshot_janus_baseline_iter_401000.caffemodel.h5 -gpu 0
