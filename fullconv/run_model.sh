#!/bin/bash

#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --gres=gpu

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh

#python surgery.py > surgery.out
python ./solve.py > solve.out
#./caffe/build/tools/caffe train -solver solver.prototxt -weights   -gpu 0
