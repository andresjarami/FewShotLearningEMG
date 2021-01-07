#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=120GB
#PBS -l walltime=15:00:00
#PBS -l software=python
#PBS -l wd

folder="Results/"
database="Cote"
windowSize="295"
for j in {1..3}
do
python3 mainDatabasesExp1.py $j 1 17 $folder $database 0 $windowSize > Cote$PBS_JOBID.log &
done
wait
