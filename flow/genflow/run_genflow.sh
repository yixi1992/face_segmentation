#!/bin/bash
#SBATCH -t 99:00:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=8
#SBATCH --partition=scavenger
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load matlab

export PBS_NODEFILE=`/usr/local/slurm/bin/generate_pbs_nodefile`

date
hostname
pwd

echo 'job id: ' $SLURM_JOBID
echo $SLURM_SUBMIT_DIR
echo $SLURM_ARRAY_TASK_ID

date
i=0
n=`wc -l $PBS_NODEFILE | cut -f1 -d' '`;
echo 'total: ' $n
for node in `cat $PBS_NODEFILE`; do
    echo $i
    cmd="module load matlab; cd $SLURM_SUBMIT_DIR; matlab -nosplash -nodesktop -nojvm -r 'genflow($i,$n)'"
    ssh $node $cmd &
    (( i += 1 ))
done
wait
date
