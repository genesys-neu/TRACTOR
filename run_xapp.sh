#!/bin/bash
# TODO add arguments
# ARG 1: model path
# ARG 2: normalization parameters .pkl path
# ARG 3: model type, choices=['CNN', 'Tv1', 'Tv2', 'ViT']

# release previously opened sockets
kill -9 `pidof python3`

# Run agent, sleep, run connector
echo "[`date`] Run xApp" > /home/container.log
cd /home/sample-xapp/TRACTOR && python3 run_xapp.py --model_path $1 --norm_param_path $2 --model_type $3 &

echo "[`date`] Pause 10 s" >> /home/container.log
sleep 10

echo "[`date`] Run connector" >> /home/container.log
cd /home/xapp-sm-connector && ./run_xapp.sh
