#!/bin/bash

#SBATCH --job-name=buta_t0
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="/home/boittier/CubeTools/charges/f-butadiene/21_eq.xyz" 
cube="/home/boittier/CubeTools/cubes/f-butadiene/B.0.p.cube"
in_pickle="False"
out_pickle="f-butadiene-0.p"
eps=1.5e-5
maxiter=5


echo $charges >> $out_pickle.out
echo $cube >> $out_pickle.out
echo $in_pickle >> $out_pickle.out
echo $out_pickle >> $out_pickle.out
echo $eps >> $out_pickle.out
echo $maxiter >> $out_pickle.out

date >> $out_pickle.out
NUMBA_NUM_THREADS=8 python ../Optimize.py $charges $cube $in_pickle $out_pickle $eps $maxiter >> $out_pickle.out
date >> $out_pickle.out
