#!/bin/bash
#SBATCH --account=windse
#SBATCH --job-name=flowers
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --output=output/flowers.%a.out
#SBATCH --mail-user=michael.locascio@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL


# go to starting directory
cd $HOME/flowers

# load conda environment
source $HOME/.bashrc
env_flowers

# get the JOB and SUBJOB ID
if [[ $SLURM_ARRAY_JOB_ID ]] ; then
	export JOB_ID=$SLURM_ARRAY_JOB_ID
	export SUB_ID=$SLURM_ARRAY_TASK_ID
else
	export JOB_ID=$SLURM_JOB_ID
	export SUB_ID=1
fi

# Run our job
srun --unbuffered -n 1 python ./opt_multistart.py flowers analytical small $SUB_ID #> solutions/multi.$SUB_ID.log 2>&1

# submit as follows
# $ sbatch --array=0-49 input/flowers_array.sh