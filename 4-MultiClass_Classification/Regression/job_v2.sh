#!/bin/sh
#SBATCH --output=simple.out 
#SBATCH --error=simple.err 
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem 4G
#SBATCH --gres=gpu:4
#SBATCH -p long
module load python/gcc-4.8.5/3.6.3
python boston_baseline.py
