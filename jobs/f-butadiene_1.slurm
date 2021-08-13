#!/bin/bash

#SBATCH --job-name=buta_t0.2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="/home/boittier/CubeTools/charges/f-butadiene/21_eq.xyz" 
cube="/home/boittier/CubeTools/cubes/f-butadiene/B.1.p.cube"
in_pickle="/home/boittier/CubeTools/pickles/f-butadiene-1.p"
#in_pickle="False"
out_pickle_name="f-butadiene-1.p"
out_pickle="/home/boittier/CubeTools/pickles/"$out_pickle_name
eps=1.5e-8
maxiter=100
ignoreIndices="0_1_4_6_7_15"
log="/home/boittier/CubeTools/logs/"$out_pickle_name.log
optimize_py="/home/boittier/CubeTools/bin/Optimize.py"

echo "Charges:    "$charges >> $log
echo "Cube file:  "$cube >> $log
echo "In pickle:  "$in_pickle >> $log
echo "Out pickle: "$out_pickle >> $log
echo "EPS (BFGS): "$eps >> $log
echo "Maxiter:    "$maxiter >> $log
echo "Ignore:     "$ignoreIndices >> $log

date >> $log
NUMBA_NUM_THREADS=8 python $optimize_py $charges $cube $in_pickle $out_pickle $eps $maxiter $ignoreIndices >> $log
date >> $log
echo "" >> $log
