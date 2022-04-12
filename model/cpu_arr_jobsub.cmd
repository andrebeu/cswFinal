#!/usr/bin/env bash

#SBATCH -t 5:59:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 2         # number of cores 
#SBATCH -o ./slurms/output.%j.%a.out

## scotty
conda init bash
conda activate sem

## submit job
srun python gs0317.py 

echo "done.sh"

# slurm diagnostics
sacct --format="CPUTime,MaxRSS"

