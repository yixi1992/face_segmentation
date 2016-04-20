#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --partition=scavenger
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="camvidconvert"


source ./load_deeplab_dependencies.sh
python convert.py
