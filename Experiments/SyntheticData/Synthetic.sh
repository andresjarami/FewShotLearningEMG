#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=40GB
#PBS -l walltime=1:00:00
#PBS -l software=python
#PBS -l wd


for per in {0..4}
do
	for t in {0..99}
	do
		python3 SyntheticData2.py 2 2 1 400 100 50 0 0 1 100 $per $t  > Sys$PBS_JOBID.log &

	done
done
wait
