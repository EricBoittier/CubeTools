#!/bin/bash

#SBATCH --job-name=butadi_4
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

hostname  # in case the job crashes we know where it was
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Numba

charges="charges/butadiene/13-charges/13_charges_refined.xyz" 
cube="cubes/butadiene/scan_extract_4.xyz.chk.fchk.pot.cube"
in_pickle="butadiene_4.p"
out_pickle="butadiene_4.p"
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

