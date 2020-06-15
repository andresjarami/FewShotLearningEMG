#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=105GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/noPCA_EPN"
database="EPN"
for j in {1..3}
do

	python3 main.py $j 1 60 $folder $database 1 0 > Cote$PBS_JOBID.log &
	
done
wait
