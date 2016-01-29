# example usage: ./run_gpu exp_name port subg_index use_distance
cd dqn;
python pyserver.py $2 &
cd ..;
./run_gpu montezuma_revenge $1 $2 $3 $4;