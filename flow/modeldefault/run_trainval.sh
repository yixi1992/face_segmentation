#!/bin/bash

#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="camvidrgb"

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh
source ../../load_deeplab_dependencies.sh

python solve_trainval.py
#./caffe/build/tools/caffe train -solver solver_trainval.prototxt -weights /lustre/yixi/face_segmentation_finetune/fullconv/VGG16fc.caffemodel -gpu 0 -iterations 16000
