#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=25
#PBS -l mem=128GB
#PBS -l walltime=24:00:00
#PBS -l software=python
#PBS -l wd
 
folder="ResultsExperiment/CollectedNew"
for j in {1..3}
do

	for i in {1..60}
	do
		python3 CollectedData_Arg_SegTEST4.py $j $i $i $folder > CollectedNew$PBS_JOBID.log &
	done
done
wait
