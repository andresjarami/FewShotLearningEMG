#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=50GB
#PBS -l walltime=18:00:00
#PBS -l software=python
#PBS -l wd

folder="results17/"
database="Nina3"
windowSize="295"
for j in {1..3}
do
for i in {1..9}
do
python3 mainDatabasesExp1.py $j $i $i $folder $database 0 $windowSize > Nina3$PBS_JOBID.log &
done
done
wait
