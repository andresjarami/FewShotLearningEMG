#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=20
#PBS -l mem=30GB
#PBS -l walltime=10:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/noPCA_Cote"
database="Cote"
for j in {1..3}
do
	python3 main.py $j 1 17 $folder $database 1 0 > Cote$PBS_JOBID.log &
	
done
wait
