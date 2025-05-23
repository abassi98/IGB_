#!/bin/bash

#SBATCH --job-name=IGB
#SBATCH --nodes=15
#SBATCH --ntasks=165
#SBATCH --cpus-per-task=1
#SBATCH --time=00-01:00:00
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
n_proc=$((${8}*${9}*${11}))
mpirun -np $n_proc python main_grads.py --max_depth=$1 --width=$2 --act_func=$3 --Vw_max=$4 --Vw_min=$5 --Vb_max=$6 --Vb_min=$7 --n_w=$8 --n_b=$9 --n_net_samples=${10} --n_net_processes=${11} --n_data_samples=${12}
  
