#!/bin/bash
#PBS -P iu60
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=00:20:00
#PBS -l storage=gdata/ub7+gdata/ma05
#PBS -l wd

module load python3/3.7.4

python3 train.py  --n_threads 4  --batch_size 16 --n_resgroups 10 --n_resblocks 20 --patch_size 192 --pre_train ./model/RCAN_BIX4.pt --zg --tasmax
