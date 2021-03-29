#!/bin/bash

#SBATCH --job-name=butadi_1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="charges/butadiene/13-charges/13_charges_refined.xyz" 
cube="cubes/butadiene/scan_extract_1.xyz.chk.fchk.pot.cube"
in_pickle="butadiene_1.p"
out_pickle="butadiene_1.p"
eps=1.5e-8
maxiter=500


echo $charges >> $out_pickle.out
echo $cube >> $out_pickle.out
echo $in_pickle >> $out_pickle.out
echo $out_pickle >> $out_pickle.out
echo $eps >> $out_pickle.out
echo $maxiter >> $out_pickle.out


cd ..
NUMBA_NUM_THREADS=8 python Optimize.py $charges $cube $in_pickle $out_pickle $eps $maxiter >> $out_pickle.out

