# example usage: ./run_gpu exp_name port subg_index use_distance <eps_endt> <lr>
cd synthetic;
rl_glue &
python dqn_agent.py $2 &
python synthetic_environment.py &
python synthetic_experiment.py $2 &
cd ..;
./run_gpu montezuma_revenge $1 $2 $3 $4 $5 $6;
