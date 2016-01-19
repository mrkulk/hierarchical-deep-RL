#!/bin/bash
#SBATCH --job-name=RL_breakout_exp2_1
#SBATCH --output=slurm_logs/RL_breakout_exp2_1.out
#SBATCH --error=slurm_logs/RL_breakout_exp2_1.err
./run_gpu breakout exp2 1