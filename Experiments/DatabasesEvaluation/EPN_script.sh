#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=105GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/EPN"
database="EPN"
for j in {1..3}
do

	for i in {1..60}
	do
		python3 main.py $j $i $i $folder $database 0 0 > Cote$PBS_JOBID.log &
	done
done
wait
