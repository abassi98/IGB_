#!/bin/bash

#SBATCH --job-name=IGB
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --time=00-05:00:00
#SBATCH --partition=earth-3
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=igb.out
#SBATCH --error=igb.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0 	# comment to run on your machine
# the above code is meant to run on ZAHW cluster in Zurich. Modify it to use in your machine/cluster. 
# from here onwards you can run everywhere

#conda activate hydro
conda activate igb
n_proc=$((${7}*${8}))
mpirun -np $n_proc python main_dip.py --max_depth=$1 --act_func=$2 --Vw_max=$3 --Vw_min=$4 --Vb_max=$5 --Vb_min=$6 --n_w=$7 --n_b=$8 --n_samples=$9 
  
