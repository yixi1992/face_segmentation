#!/bin/bash

#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --job-name="camvidf1b1"
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh
source ../../load_deeplab_dependencies.sh

python ./solve.py > solve.out
