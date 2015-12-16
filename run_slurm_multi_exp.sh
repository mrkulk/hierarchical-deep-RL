jobname='mz_test'

for  seed in 6000; do
     for subgoal in {2..12}; do
        for usedistance in 'true' 'false'; do
                 stdOut=log.${temp}.stdout
                 stdErr=log.${temp}.stderr
                 temp="seed_${seed}_subgoal_${subgoal}_usedistance_${usedistance}"
                 resFile=result.${temp}
                 stdOut=log.${jobname}.${temp}.stdout
                 stdErr=log.${jobname}.${temp}.stderr
                 logRoot=slurm_logs


                 sbatch  ${jobRunTime}  -o ${logRoot}/${stdOut}  -e ${logRoot}/${stdErr}   --job-name=${jobname}  run_slurm.sh ${seed} ${subgoal} ${usedistance} 
                sleep 2

done
done
done

