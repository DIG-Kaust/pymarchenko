#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pylops
echo 'Activated environment:' $(which python)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python Rayleigh_Marchenko_imaging.py
