#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt


eNB_PORT=5305
UE_PORT=5315
ip=3
out_file=output_$1
out_dir=${1%.*}
num=1010123456002

echo "using $1, results will be saved in ./$out_dir"
sleep 2
mkdir $out_dir

read -r gnb < $1
echo "gnb is: $gnb"
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'

while IFS= read -r line; do
    echo "Configuring SRN: $line"
    sshpass -p "scope" scp radio_tgen.conf $line:/root/radio_api/
    sshpass -p "scope" scp ./traffic_gen.py $line:/root/traffic_gen/
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_tgen.conf" &
    sshpass -p "scope" rsync -av ../raw $line:/root/traffic_gen/
    sleep 3
done < $1

sleep 5
clear -x
echo "Configured all SRNs"
sleep 5

while IFS= read -r line; do
    if [ $line != $gnb ]
    then
    	echo "Starting TGEN for SRN: $line"
    	trace=$(ls ../raw/*.csv | shuf -n 1)
    	echo $trace
    	echo "Using trace: $trace for $num" >> ./$out_dir/$out_file
    	echo "Using ip: 172.16.0.${ip}"
    	echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
    	echo "Starting gNB"
    	sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ${trace} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
    	sleep 3
    	echo "Starting UE"
    	sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ${trace} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
    	ip=$((ip+1))
    	eNB_PORT=$((eNB_PORT+1))
    	UE_PORT=$((UE_PORT+1))
    	num=$((num+1))
    	sleep 1
    fi
done < $1

#wait
sleep 1800

sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./$out_dir/

echo "All tests complete"
kill $(jobs -p)

