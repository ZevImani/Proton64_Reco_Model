#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-09:00      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu			# Partition to submit to
#SBATCH --mem=12G          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zResNet_edep_xymag_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zResNet_edep_xymag_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 		# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

start_time=$(date)
echo "Start Time: $(date)"

conda run -n ldm python3 -u reco_xymag.py 

echo "End Time: $(date)"

echo "Elapsed time: $((($(date +%s) - $(date -d "$start_time" +%s))/60)) minutes"

