#!/bin/bash

#SBATCH --job-name=B_7-trans
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="charges/b/trans/25_charges_refined_b_tr.xyz" 
cube="cubes/b/B.7.p.cube"
in_pickle="False"
out_pickle="b_7_tr.p"
eps=1.5e-6
maxiter=100000

log=b_jobs/$out_pickle.log


cd ..


hostname > $log
echo $charges > $log
echo $cube > $log
echo $out_pickle > $log

NUMBA_NUM_THREADS=8 python Optimize.py $charges $cube $in_pickle $out_pickle $eps $maxiter > $log


