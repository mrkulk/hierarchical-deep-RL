#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p general
#SBATCH --mem 70000
#SBATCH -t 5-23:00
#SBATCH --mail-user=tejask@mit.edu

cd dqn
/home/tejask/envs/my_root/bin/python pyserver.py &
cd ..
./run_gpu montezuma_revenge fullrun1 5550 12 false
