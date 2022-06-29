#!/bin/bash
#SBATCH --account=windse
#SBATCH --time=16:00:00
#SBATCH --job-name=multi
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
##SBATCH --partition=debug
#SBATCH --mail-user=michael.locascio@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=output/multi_sol.%a.out

# go to starting directory
cd $HOME/flowers

# load conda environment
env_flowers()

# get the JOB and SUBJOB ID
if [[ $SLURM_ARRAY_JOB_ID ]] ; then
	export JOB_ID=$SLURM_ARRAY_JOB_ID
	export SUB_ID=$SLURM_ARRAY_TASK_ID
else
	export JOB_ID=$SLURM_JOB_ID
	export SUB_ID=1
fi

# Run our job (?)
srun -n 36 python ./multistart.py $SUB_ID > output/multi$SUB_ID.log 2>&1

# submit as follows
# $ sbatch --array=0-100 -N1 array.sh