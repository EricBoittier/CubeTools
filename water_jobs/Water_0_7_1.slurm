#!/bin/bash

#SBATCH --job-name=Water_0_7_1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="/home/boittier/water/6_charges_refined.xyz" 
cube="/home/boittier/water/Water_0_7_1.pot.cube"
in_pickle="Water_OPT"
out_pickle="Water_0_7_1_out.p"
eps=1.5e-7
maxiter=100


echo $charges
echo $cube
echo $in_pickle
echo $out_pickle
echo $eps
echo $maxiter


cd ..
NUMBA_NUM_THREADS=8 python Optimize.py $charges $cube $in_pickle $out_pickle $eps $maxiter > $out_pickle.out
