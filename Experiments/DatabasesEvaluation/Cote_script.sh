#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=120GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/Cote"
database="Cote"
for j in {1..3}
do
	for i in {1..17}
	do
		python3 main.py $j $i $i $folder $database 1 0 0 1 > Cote$PBS_JOBID.log &
	done
done
wait
