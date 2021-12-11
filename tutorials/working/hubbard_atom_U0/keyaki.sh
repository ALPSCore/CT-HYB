#!/bin/bash
#SBATCH -p defq
#SBATCH -n 14
#SBATCH --ntasks-per-node=14
#SBATCH -J alpscthyb
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

python3 mk_model.py
date > output
mpirun -np 14 hybmat input.ini > output
date >> output
python3 plot.py
