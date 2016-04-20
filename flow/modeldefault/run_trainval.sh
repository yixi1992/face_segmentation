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

./caffe/build/tools/caffe train -solver solver_trainval.prototxt -weights snapshots_tvtcamvidfmp200200/train_lr1e-10/_iter_16000.caffemodel -gpu 0
