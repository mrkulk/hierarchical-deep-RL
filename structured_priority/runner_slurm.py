#run script using slurm
import os

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

# Don't give it a `save` name - that gets generated for you
jobs = [
  [
    'breakout', #game 
    'exp1', #game name
    0 #priority on/off
  ],
  [
    'breakout',
    'exp1',
    1
  ],
]

for jj in range(len(jobs)):
    jobname = "RL"
    flagstring = ""
    for ii in range(len(jobs[jj])):
      jobname = jobname + "_" + str(jobs[jj][ii])
      flagstring = flagstring + " " + str(jobs[jj][ii])


    if not os.path.exists("slurm_logs/" + jobname):
        os.makedirs("slurm_logs/" + jobname)

    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("./run_gpu" + flagstring)

    print ("./run_gpu" + flagstring)
    if False:
        os.system("sbatch --qos=cbmm --mem=40000 -N 1 -c 2 --gres=gpu:1 --time=5-00:00:00 slurm_scripts/" + jobname + ".slurm &")




