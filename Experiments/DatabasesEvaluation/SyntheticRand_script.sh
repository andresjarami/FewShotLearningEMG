#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=58GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -l wd
 
folder="ResultsExp2/"
for i in {0,1,3,5,10,15,20}
do	
	for j in {0..99}
	do
		python3 mainSyntheticExp2.py $i $folder $j > Synthetic$PBS_JOBID.log &
	done
done
wait
