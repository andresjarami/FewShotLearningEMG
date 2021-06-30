#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=50GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/"
database="Capgmyo_dba"
windowSize="100"
for j in {1..3}
do
for i in {1..18}
do
python3 mainDatabasesExp1.py $j $i $i $folder $database 0 $windowSize > CapgmyoA$PBS_JOBID.log &
done
done
wait
