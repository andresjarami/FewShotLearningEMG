#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=35
#PBS -l mem=60GB
#PBS -l walltime=15:00:00
#PBS -l software=python
#PBS -l wd

folder="Results/"
database="Nina5"
windowSize="295"
for j in {1..3}
do
python3 mainDatabasesExp1.py $j 1 10 $folder $database 0 $windowSize > Nina$PBS_JOBID.log &
done
wait
