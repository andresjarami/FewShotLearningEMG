#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=20
#PBS -l mem=12GB
#PBS -l walltime=10:00:00
#PBS -l software=python
#PBS -l wd
 
folder="ResultsExperiment/NinaNew"
for j in {1..3}
do

	for i in {1..10}
	do
		python3 Nina5_Arg_SegTEST.py $j $i $i $folder > NinaNew$PBS_JOBID.log &
	done
done
wait
