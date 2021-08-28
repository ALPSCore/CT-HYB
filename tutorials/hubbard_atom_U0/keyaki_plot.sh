#!/bin/bash
#SBATCH -p defq
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -J alpscthyb
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

echo $SLURM_CPUS_ON_NODE > output-plot
mpirun -np $SLURM_CPUS_ON_NODE python3 plot.py >> output-plot
