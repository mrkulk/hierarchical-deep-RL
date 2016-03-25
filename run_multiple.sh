# example usage: ./run_multiple.sh START_PORT exp_name port subg_index use_distance <eps_endt> dqn_TrueorFalse
START_PORT=$1
for port in 10 20 30 40 50; do 
    port=$((START_PORT+port))
    rlport=$((port+5))
    cd synthetic;
    RLGLUE_PORT=$rlport rl_glue &
    RLGLUE_PORT=$rlport python dqn_agent.py $port &
    RLGLUE_PORT=$rlport python synthetic_environment_hard.py &
    RLGLUE_PORT=$rlport python synthetic_experiment.py $port &
    cd ..;
    ./run_gpu montezuma_revenge $2.$port $port $3 $4 $5 $6 $7 & 
    sleep 1;
done;

