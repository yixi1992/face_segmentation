#!/bin/bash

#SBATCH -t 10:00:00

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh

python eval.py > eval.out
#./caffe/build/tools/caffe train -solver face_segmentation_finetune_solver.prototxt -weights ../image+jit2d+illum+rot3d/snapshots/snapshot_janus_baseline_iter_401000.caffemodel.h5 -gpu 0
