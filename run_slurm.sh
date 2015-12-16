#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p general
#SBATCH --mem 50000
#SBATCH -t 1-23:00
#SBATCH --mail-user=av.saeedi@gmail.com

cd dqn
/home/tejask/envs/my_root/bin/python pyserver.py &
cd ..
./run_gpu montezuma_revenge 'subgoal_'$2'_usedist_'$3 $((5000+$num2)) --seed $1 --subgoal $2  --usedistance $3;