#!/bin/bash
#SBATCH -p defq
#SBATCH -n 64
#SBATCH --ntasks-per-node=64
#SBATCH -J alpscthyb
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

date > output
mpirun -np 64 hybmat input.ini > output
date >> output
