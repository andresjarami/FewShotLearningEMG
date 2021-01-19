#!/bin/bash
#PBS -P oj72
#PBS -q hugemem
#PBS -l ncpus=48
#PBS -l mem=440GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/"
database="EPN"
windowSize="295"
for j in {1..3}
do
python3 mainDatabasesExp1.py $j 1 30 $folder $database 0 $windowSize > EPN$PBS_JOBID.log &
done
wait
