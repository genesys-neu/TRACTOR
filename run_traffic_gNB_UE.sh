#!/bin/bash
for t in ./raw/*.csv
do
  echo "***** Run traffic on gNB *****"
  sshpass -p "scope" ssh $1 "cd /root/traffic_gen && python traffic_gen.py --eNB -f ${t}" &   # this will have to let the next command go through
  echo "Sleep for 3 secs"
  sleep 3  # let's wait few seconds
  echo "***** Run traffic on UE *****"
  sshpass -p "scope" ssh $2 "cd /root/traffic_gen && python traffic_gen.py -f ${t}"
  #wait  # this will wait until all child processes terminates
  echo "***** Copy data *****"
  folder=$(dirname ${t})
  tracename=$(basename ${t})
  filename="${tracename%.*}"
  sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 mkdir -p /home/${folder}/${filename}"
  sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 sh -c 'mv /home/*.pkl /home/${folder}/${filename}/.'"
  sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 sh -c 'mv /home/*.log /home/${folder}/${filename}/.'"
done