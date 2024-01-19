#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 8 UEs
## Then there should be the interferer
## Then there should be the observer
## Then there should be the near-RT RIC


eNB_PORT=5305
UE_PORT=5315
ip=3
out_file=output_$1
out_dir=${1%.*}
num=1010123456002


echo "using $1, results will be saved in ./$out_dir"
sleep 10
mkdir $out_dir

read -r gnb < $1
echo "gnb is: $gnb"
# Start the channel
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'
# remove any existing metrics
sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"

interferer=$(sed '10!d' $1)
sshpass -p "sunflower" scp uhd_tx_tone.sh $interferer:/root/utils/

listener=$(sed '11!d' $1)
# TODO add listener configuration here

ric=$(sed '12!d' $1)

# start the gNB and UEs
while IFS= read -r line; do
    if [ $line != $interferer ] && [ $line != $listener ] && [ $line != $ric ]
    then
      echo "Configuring SRN: $line"
      sshpass -p "scope" scp radio_IMPACT.conf $line:/root/radio_api/
      sshpass -p "scope" scp ../traffic_gen.py $line:/root/traffic_gen/
      sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_IMPACT.conf" &
      sshpass -p "scope" rsync -av ../raw $line:/root/traffic_gen/
      sleep 5
    fi
done < $1

# Start the near-RT RIC
sshpass -p "ChangeMe" ssh $ric 'cd ~ && cd radio_code/colosseum-near-rt-ric/setup-scripts/ && ./setup-ric.sh col0'

# connect the gNB and RIC
IPCOL0=`sshpass -p "ChangeMe" ssh $ric 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo $IPCOL0
sshpass -p "scope" scp ../colosseum/radio_code/colosseum-scope-e2/src/du_app/csv_reader.* $gnb:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp ../colosseum/radio_code/colosseum-scope-e2/src/du_app/readLastMetrics.* $gnb:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp ../colosseum/radio_code/colosseum-scope-e2/src/du_app/bs_connector.* $gnb:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp ../colosseum/radio_code/colosseum-scope-e2/build/odu/makefile $gnb:/root/radio_code/colosseum-scope-e2/build/odu/.
sshpass -p "scope" scp ../colosseum/radio_code/colosseum-scope-e2/run_odu.sh $gnb:/root/radio_code/colosseum-scope-e2/run_odu.sh
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/src/du_app/ && g++ readLastMetrics.cpp -o readLastMetrics.o"
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && sed -i 's/172.30.199.202/${IPCOL0}/' build_odu.sh && ./build_odu.sh clean" # && ./run_odu.sh

sleep 10
# Start the ODU
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && ./run_odu.sh"

echo "The gnb ID is: "
sshpass -p "ChangeMe" ssh $ric "docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1"
sleep 10

#TODO setup_TRACTOR_RIC.sh

sleep 30
clear -x
echo "Configured all SRNs"
sleep 30


echo "Starting TGEN for demo UE"
tracename=IMPACT.csv
echo "Using trace: ${tracename}"
line=$(sed '2!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2

echo "Starting TGEN for eMBB 1"
tracename=embb_11_18.csv
echo "Using trace: ${tracename}"
line=$(sed '3!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for eMBB 2"
tracename=embb_03_03a.csv
#ip=5
echo "Using trace: ${tracename}"
line=$(sed '4!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for eMBB 3"
tracename=embb_04_10.csv
#ip=6
echo "Using trace: ${tracename}"
line=$(sed '5!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for eMBB 4"
tracename=embb_06_09.csv
#ip=7
echo "Using trace: ${tracename}"
line=$(sed '6!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for urllc 1"
tracename=urllc_05_18.csv
#ip=8
echo "Using trace: ${tracename}"
line=$(sed '7!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for urllc 2"
tracename=urllc_06_12.csv
#ip=9
echo "Using trace: ${tracename}"
line=$(sed '8!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for mmtc 1"
tracename=mmtc_05_18.csv
echo "Using trace: ${tracename}"
line=$(sed '9!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# approximately 32s into experiment
# wait
sleep 88 
# start the interference
sshpass -p "sunflower" ssh $2 "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $2 "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $2 "kill -INT ${int_PID}"

# approximately 240s into experiment
echo "Starting TGEN for urllc 1"
tracename=urllc_05_18.csv
ip=8
echo "Using trace: ${tracename}"
line=$(sed '7!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2

echo "Starting TGEN for urllc 2"
tracename=urllc_06_12.csv
ip=9
echo "Using trace: ${tracename}"
line=$(sed '8!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# wait
sleep 52
sshpass -p "sunflower" ssh $2 "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $2 "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 50
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $2 "kill -INT ${int_PID}"

# approximately 360s into experiment
echo "Starting TGEN for eMBB 3"
tracename=embb_04_10.csv
ip=6
echo "Using trace: ${tracename}"
line=$(sed '5!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


echo "Starting TGEN for eMBB 4"
tracename=embb_06_09.csv
ip=7
echo "Using trace: ${tracename}"
line=$(sed '6!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


#wait
sleep 112
sshpass -p "sunflower" ssh $2 "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $2 "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $2 "kill -INT ${int_PID}"

# approximately 600s into experiment
echo "Starting TGEN for eMBB 2"
tracename=embb_03_03a.csv
ip=5
echo "Using trace: ${tracename}"
line=$(sed '4!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2

echo "Starting TGEN for urllc 1"
tracename=urllc_05_18.csv
ip=8
echo "Using trace: ${tracename}"
line=$(sed '7!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2

echo "Starting TGEN for urllc 2"
tracename=urllc_06_12.csv
ip=9
echo "Using trace: ${tracename}"
line=$(sed '8!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd traffic_gen && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd traffic_gen && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# wait
sleep 48
sshpass -p "sunflower" ssh $2 "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $2 "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $2 "kill -INT ${int_PID}"


sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./$out_dir/

echo "All tests complete"
kill $(jobs -p)

