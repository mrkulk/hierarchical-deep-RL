#!/bin/bash
#SBATCH --job-name=RL_breakout_exp1_1
#SBATCH --output=slurm_logs/RL_breakout_exp1_1.out
#SBATCH --error=slurm_logs/RL_breakout_exp1_1.err
./run_gpu breakout exp1 1