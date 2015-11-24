#!/bin/bash
#SBATCH --job-name=DQN
#SBATCH --output=slurm_logs/DQN.out
#SBATCH --error=slurm_logs/DQN.err
/home/tejask/envs/my_root/bin/python dqn/pyserver.py &
./run_gpu montezuma_revenge
