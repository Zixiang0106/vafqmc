#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 40:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
source ~/init.sh
conda activate vafqmc
srun python -u run.py > out.txt 2> >(cat >&2)
