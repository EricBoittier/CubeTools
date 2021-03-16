#!/bin/bash

#SBATCH --job-name=refine-16chg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=long

cd /home/boittier/MDCM-1/examples/butadiene/6-refine-simplex/refined/16-charges

/home/boittier/MDCM-1/bin/pcubefit.x -xyz /home/boittier/MDCM-1/examples/butadiene/5-combine-frags/16-combined.xyz /home/boittier/MDCM-1/examples/butadiene/2-mtp-fit/fitted-mtpl.dat -simplex -esp /data/boittier/gaussian_scans/butadiene/scan_extract_0.xyz.chk.fchk.pot.cube -dens /data/boittier/gaussian_scans/butadiene/scan_extract_0.xyz.chk.fchk.den.cube -nacmax 4 -ntry 1 -v > 16-charges.out

