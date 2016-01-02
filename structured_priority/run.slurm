#!/bin/bash
#SBATCH --job-name=DQN
#SBATCH --output=slurm_logs/DQN.out
#SBATCH --error=slurm_logs/DQN.err
./run_gpu breakout
