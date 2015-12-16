#!/bin/bash
#SBATCH --job-name=DQN
#SBATCH --output=slurm_logs/DQN.out
#SBATCH --error=slurm_logs/DQN.err
#cd dqn
#/home/tejask/envs/my_root/bin/python pyserver.py &
#cd ..
#./run_gpu montezuma_revenge

./run_exp.sh test 5000
