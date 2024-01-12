#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt genesys-INT 
## Ensure the gNB is the first SRN in config_file.txt


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
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'
sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"
sshpass -p "sunflower" scp uhd_tx_tone.sh $2:/root/utils/

while IFS= read -r line; do
    echo "Configuring SRN: $line"
    sshpass -p "scope" scp radio_IMPACT_baseline.conf $line:/root/radio_api/
    sshpass -p "scope" scp ../traffic_gen.py $line:/root/traffic_gen/
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_IMPACT_baseline.conf" &
    sshpass -p "scope" rsync -av ../raw $line:/root/traffic_gen/
    sleep 5
done < $1

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

