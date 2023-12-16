#!/bin/bash
# args:
# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id

sshpass -p "scope" ssh $1 "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"

for t in ./raw/*.csv
do
  echo "TRACE DIR $t"
  echo "***** Run traffic on gNB *****"
  sshpass -p "scope" ssh -tt $1 "cd /root/TRACTOR && python traffic_gen.py --eNB -f ${t}" &   # this will have to let the next command go through
  echo "Sleep for 5 secs"
  sleep 5  # let's wait few seconds
  echo "***** Run traffic on UE *****"
  start_ts=`date +%s%N | cut -b1-13`
  sshpass -p "scope" ssh -tt $2 "cd /root/TRACTOR && python traffic_gen.py -f ${t}" &
  wait  # this will wait until all child processes terminates
  end_ts=`date +%s%N | cut -b1-13`
  echo "START: $start_ts\nEND: $end_ts\n"
  echo "$start_ts,$end_ts" > ${t}_se_info.out
  echo "***** Sleeping... *****"
  sleep 5 # sleep for a few second to allow all the classifier outputs to complete producing files
  echo "***** Copy data *****"
  folder=$(dirname ${t})
  tracename=$(basename ${t})
  filename="${tracename%.*}"
  target_dir="/home/${folder}/${filename}/."
  source_dir="/home"
  sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 mkdir -p ${target_dir}"
  sshpass -p "ChangeMe" ssh $3 "docker exec sample-xapp-24 /home/mv_ts_files.sh ${target_dir} ${source_dir} ${start_ts} ${end_ts}"
  sshpass -p "ChangeMe" ssh $3 "docker cp sample-xapp-24:/home/raw /root/."
  sshpass -p "ChangeMe" rsync -av -e ssh $3:/root/raw colosseum/.
  sshpass -p "scope" scp $1:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./colosseum/raw/${tracename}_metrics.csv
done

sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 mkdir -p ${source_dir}/raw/no_active"
sshpass -p "ChangeMe" ssh $3 "docker exec -d sample-xapp-24 bash -c 'mv ${source_dir}/*.pkl ${source_dir}/no_active/.'"
sshpass -p "ChangeMe" ssh $3 "docker cp sample-xapp-24:/home/raw /root/."
sshpass -p "ChangeMe" rsync -av -e ssh $3:/root/raw colosseum/.
