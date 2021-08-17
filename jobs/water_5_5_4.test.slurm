#!/bin/bash

#SBATCH --job-name=554.0
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="/home/unibas/boittier/pydcm-1.2/water_5_5_4/5-refine/8-charges/8_charges_refined.xyz" 
cube="/home/unibas/boittier/water/Water_5_5_4.den.cube"
#in_pickle="/home/boittier/CubeTools/jobs/f-butadiene-0.2.p"
in_pickle="False"
out_pickle_name="water_5_5_4.0.p"
out_pickle="/home/unibas/boittier/CubeTools/jobs/"$out_pickle_name
eps=1.5e-7
maxiter=1
ignoreIndices="8_8"
log="/home/unibas/boittier/CubeTools/logs/"$out_pickle_name.log
optimize_py="/home/unibas/boittier/CubeTools/bin/Optimize.py"

echo "Charges:    "$charges >> $log
echo "Cube file:  "$cube >> $log
echo "In pickle:  "$in_pickle >> $log
echo "Out pickle: "$out_pickle >> $log
echo "EPS (BFGS): "$eps >> $log
echo "Maxiter:    "$maxiter >> $log
echo "Ignore:     "$ignoreIndices >> $log

date >> $log
#NUMBA_NUM_THREADS=8 python $optimize_py $charges $cube $in_pickle $out_pickle $eps $maxiter >> $log
NUMBA_NUM_THREADS=8 python $optimize_py $charges $cube $in_pickle $out_pickle $eps $maxiter $ignoreIndices >> $log
date >> $log
echo "" >> $log
