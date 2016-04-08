#!/bin/bash

#source /lustre/yixi/decouplednet/DecoupledNet/inference/load_deeplab_dependencies.sh

module load matlab
matlab -nodisplay -nosplash -r "run('genflow.m'); exit"
