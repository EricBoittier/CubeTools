#!/bin/bash

#SBATCH --job-name=Water_4_4_3
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="/home/boittier/water/6_charges_refined.xyz" 
cube="/home/boittier/water/Water_4_4_3.pot.cube"
#in_pickle="False"
#in_pickle="/home/boittier/CubeTools/water_results/Water_OPT"
out_pickle_name="Water_4_4_3_out.p"
in_pickle="/home/boittier/CubeTools/water_results/Water_4_4_4_out.1.p"
out_pickle="/home/boittier/CubeTools/water_results/"$out_pickle_name
eps=1.5e-8
maxiter=100
ignoreIndices="0"
log="/home/boittier/CubeTools/logs/"$out_pickle_name.log
optimize_py="/home/boittier/CubeTools/bin/Optimize.py"

echo "Charges:    "$charges >> $log
echo "Cube file:  "$cube >> $log
echo "In pickle:  "$in_pickle >> $log
echo "Out pickle: "$out_pickle >> $log
echo "EPS (BFGS): "$eps >> $log
echo "Maxiter:    "$maxiter >> $log
echo "Ignore:     $ignoreIndices" >> $log

date >> $log
NUMBA_NUM_THREADS=8 python $optimize_py $charges $cube $in_pickle $out_pickle $eps $maxiter $ignoreIndices >> $log
date >> $log
echo "" >> $log
