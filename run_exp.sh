if [ -z "$1" ]
  then echo "Please provide the logname and port for running the experiment e.g.  ./run_exp basic1 5000 "; exit 0
fi
cd dqn;
python pyserver.py $2 &
cd ..;
./run_gpu montezuma_revenge $1 $2;