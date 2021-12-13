#!/bin/bash
#SBATCH -p defq
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH -J alpscthyb
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

python3 mk_model.py
date > output
mpirun -np 16 hybmat input.ini > output
date >> output
#python3 plot.py
