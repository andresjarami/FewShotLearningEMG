#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=35
#PBS -l mem=60GB
#PBS -l walltime=15:00:00
#PBS -l software=python
#PBS -l wd

folder="Results/NinaPro5"
database="Nina5"
for j in {1..3}
do

	for i in {1..10}
	do
		python3 mainDatabasesExp1.py $j $i $i $folder $database 1 0 0 1 > Nina$PBS_JOBID.log &
	done
done
wait
