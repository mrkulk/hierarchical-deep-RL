#!/bin/bash
#SBATCH --job-name=RL_breakout_exp2_0
#SBATCH --output=slurm_logs/RL_breakout_exp2_0.out
#SBATCH --error=slurm_logs/RL_breakout_exp2_0.err
./run_gpu breakout exp2 0