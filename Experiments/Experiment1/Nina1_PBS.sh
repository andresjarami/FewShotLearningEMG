#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=70GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/"
database="Nina1"
windowSize="280"
for j in {1..3}
do
for i in {1..27}
do
python3 mainDatabasesExp1.py $j $i $i $folder $database 0 $windowSize > Nina1$PBS_JOBID.log &
done
done
wait
