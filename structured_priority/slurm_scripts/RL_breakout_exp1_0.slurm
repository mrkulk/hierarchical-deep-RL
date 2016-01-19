#!/bin/bash
#SBATCH --job-name=RL_breakout_exp1_0
#SBATCH --output=slurm_logs/RL_breakout_exp1_0.out
#SBATCH --error=slurm_logs/RL_breakout_exp1_0.err
./run_gpu breakout exp1 0