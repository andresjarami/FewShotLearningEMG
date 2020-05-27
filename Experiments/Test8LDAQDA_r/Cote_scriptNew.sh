#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=20
#PBS -l mem=30GB
#PBS -l walltime=10:00:00
#PBS -l software=python
#PBS -l wd
 
folder="ResultsExperiment/CoteNew"
for j in {1..3}
do

	for i in {1..17}
	do
		python3 CoteAllard_Arg_SegTEST.py $j $i $i $folder > CoteNew$PBS_JOBID.log &
	done
done
wait
