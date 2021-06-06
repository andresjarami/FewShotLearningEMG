#!/bin/bash
#PBS -P oj72
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=7GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd




python3 evaluate_wavelet_target_network.py > Pre_Training$PBS_JOBID.log 

