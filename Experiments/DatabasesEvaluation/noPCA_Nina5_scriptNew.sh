#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=25
#PBS -l mem=13GB
#PBS -l walltime=10:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/noPCA_NinaPro5"
database="Nina5"
for j in {1..3}
do

	python3 main.py $j 1 10 $folder $database 1 0 > Nina$PBS_JOBID.log &
	
done
wait
