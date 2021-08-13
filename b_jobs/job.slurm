#!/bin/bash

#SBATCH --job-name=B_1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="charges/b/equil/25_charges_refined_b_eq.xyz" 
cube="cubes/b/B.0.p.cube"
in_pickle="False"
out_pickle="b_0_eq.p"
eps=1.5e-6
maxiter=10

log=b_jobs/$out_pickle.log


cd ..


hostname > $log
echo $charges > $log
echo $cube > $log
echo $out_pickle > $log

NUMBA_NUM_THREADS=8 python Optimize.py $charges $cube $in_pickle $out_pickle $eps $maxiter > $log


