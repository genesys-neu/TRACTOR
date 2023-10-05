#!/bin/bash
## HOW TO RUN: ./interference_tgen.sh genesys-gNB genesys-UE


if [ ! -d "./interference/mal_traf/udp_flood" ]
  then
    mkdir ./interference
    mkdir ./interference/mal_traf
    mkdir ./interference/mal_traf/udp_flood
fi

sshpass -p "scope" ssh $1 "colosseumcli rf start 1017 -c"

sshpass -p "scope" scp radio_tgen.conf $1:/root/radio_api/
sshpass -p "scope" scp radio_tgen.conf $2:/root/radio_api/
sshpass -p "scope" scp traffic_gen_uf.py $1:/root/traffic_gen/
sshpass -p "scope" scp traffic_gen_uf.py $2:/root/traffic_gen/
    
sshpass -p "scope" ssh $1 "cd /root/radio_api && python3 scope_start.py --config-file radio_tgen.conf" &
sshpass -p "scope" rsync -av ./raw $1:/root/traffic_gen/

sshpass -p "scope" ssh $2 "cd /root/radio_api && python3 scope_start.py --config-file radio_tgen.conf" &
sshpass -p "scope" rsync -av ./raw $2:/root/traffic_gen/


sleep 20
clear -x
echo "Configured all SRNs"
sleep 20

for t in ./raw/*.csv
do
  tracename=$(basename ${t})
  echo "TRACE ${tracename}"
  echo "***** Run traffic on gNB *****"
  sshpass -p "scope" ssh $1 "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"
  sshpass -p "scope" ssh -tt $1 "cd /root/traffic_gen && python traffic_gen_uf.py --eNB -f ./raw/${tracename}" &   # this will have to let the next command go through
  gNB_PID=$!
  echo "Sleep for 5 secs"
  sleep 5  # let's wait few seconds
  echo "***** Run traffic on UE *****"
  sshpass -p "scope" ssh -tt $2 "cd /root/traffic_gen && python traffic_gen_uf.py -f ./raw/${tracename}" &
  UE_PID=$!
  sleep 5 # let the traffic start
    
  wait $gNB_PID # this will wait until gNB processes terminates
  wait $UE_PID # this will wait until gNB processes terminates
  
  if [ $# -eq 3 ] 
    then
      echo "***** Stopping Interference PID: ${int_PID} *****"
      sshpass -p "sunflower" ssh $3 "kill -INT ${int_PID}"
  fi

  echo "***** Sleeping... *****"
  sleep 5 # sleep for a few second to allow all the classifier outputs to complete producing files
  echo "***** Copy data *****"




  sshpass -p "scope" scp $1:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./interference/mal_traf/udp_flood/${tracename}
  
  echo "***** Completed $t Preparing for next run *****"
  sleep 5 # sleep for a few second to allow the system to settle
  clear -x
  
done


echo "All tests complete"


