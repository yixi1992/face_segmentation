#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --partition=scavenger

source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh
source ../../load_deeplab_dependencies.sh

python ./solve.py
