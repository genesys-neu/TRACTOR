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

ric=$(sed '12!d' $1)

# start the gNB and UEs
while IFS= read -r line; do
    if [ $line != $interferer ] && [ $line != $listener ] && [ $line != $ric ]
    then
      echo "Configuring SRN: $line"
      sshpass -p "scope" scp radio_IMPACT.conf $line:/root/radio_api/
      sshpass -p "scope" scp ../traffic_gen.py $line:/root/traffic_gen/
      sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_IMPACT.conf" &
      sshpass -p "scope" rsync -av -e ssh --exclude 'colosseum' --exclude '.git' --exclude 'logs' --exclude 'utils/raw' --exclude 'model' ../../TRACTOR $line:/root/.
      sleep 2
      clear -x
    fi
done < $1

echo "Starting the near-RT RIC"
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
clear -x
# Start the ODU
echo "Starting the ODU"
gnome-terminal -- bash -c "sshpass -p 'scope' ssh -t $gnb 'cd /root/radio_code/colosseum-scope-e2/ && ./run_odu.sh'; bash" &
#sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && ./run_odu.sh" &

sleep 15
clear -x

echo "Starting the Near-RT RIC"
GNBID=`sshpass -p "ChangeMe" ssh $ric "docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1"`
echo "The gnb ID is: $GNBID"
sshpass -p "ChangeMe" ssh $ric "cd /root/radio_code/colosseum-near-rt-ric/setup-scripts && ./setup-sample-xapp.sh ${GNBID}"

sleep 15

echo "Copying files to the xApp"
sshpass -p "ChangeMe" rsync -av -e ssh --exclude 'colosseum' --exclude '.git' --exclude 'logs/*UE/' --exclude 'utils/raw' --exclude 'raw' ../../TRACTOR $ric:/root/.
sshpass -p "ChangeMe" ssh $ric 'docker cp /root/TRACTOR sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $ric 'docker exec sample-xapp-24 mv /home/sample-xapp/TRACTOR/utils/run_xapp_IMPACT.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp_IMPACT.sh'

echo "Starting the xApp"
#sshpass -p "ChangeMe" ssh $ric 'docker exec -i sample-xapp-24 bash -c "rm /home/*.log && cd /home/sample-xapp/ && ./run_xapp_IMPACT.sh"' &
gnome-terminal -- bash -c "sshpass -p 'ChangeMe' ssh $ric 'docker exec -i sample-xapp-24 bash -c \"rm /home/*.log && cd /home/sample-xapp/ && ./run_xapp_IMPACT.sh\"'; bash" &

sleep 10
echo "Starting the listener"
sshpass -p "sunflower" ssh $listener "sed -i 's/--freq 1\.010e9/--freq 1.020e9/' utils/uhd_rx_fft.sh"
gnome-terminal -- bash -c "sshpass -p 'sunflower' ssh -t $listener 'sh utils/uhd_rx_fft.sh'; bash" &

sleep 20
clear -x
echo "Configured all SRNs"
sleep 20


echo "Starting TGEN for demo UE"
tracename=IMPACT.csv
echo "Using trace: ${tracename}"
line=$(sed '2!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# approximately 32s into experiment
# wait
sleep 88
echo "starting the interference"
sshpass -p "sunflower" ssh $interferer "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $interferer "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $interferer "kill -INT ${int_PID}"

# approximately 240s into experiment
echo "Starting TGEN for urllc 1"
tracename=urllc_05_18.csv
ip=8
echo "Using trace: ${tracename}"
line=$(sed '7!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# wait
sleep 52
echo "starting the interference"
sshpass -p "sunflower" ssh $interferer "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $interferer "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 50
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $interferer "kill -INT ${int_PID}"

# approximately 360s into experiment
echo "Starting TGEN for eMBB 3"
tracename=embb_04_10.csv
ip=6
echo "Using trace: ${tracename}"
line=$(sed '5!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


#wait
sleep 112
sshpass -p "sunflower" ssh $interferer "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $interferer "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $interferer "kill -INT ${int_PID}"

# approximately 600s into experiment
echo "Starting TGEN for eMBB 2"
tracename=embb_03_03a.csv
ip=5
echo "Using trace: ${tracename}"
line=$(sed '4!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
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
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))
sleep 2


# wait
sleep 48
sshpass -p "sunflower" ssh $interferer "cd /root/utils && sh uhd_tx_tone.sh" &
sleep 10 # let the tx process start
int_PID=`sshpass -p "sunflower" ssh $interferer "pgrep tx_waveforms"`
echo "****** Returned PID: ${int_PID} ***********"
sleep 110
echo "***** Stopping Interference PID: ${int_PID} *****"
sshpass -p "sunflower" ssh $interferer "kill -INT ${int_PID}"


sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./$out_dir/

#TODO: verify copy log file from xApp
sshpass - p "ChangeMe" ssh $ric "docker cp sample-xapp-24:/home/*log* /root/."
sshpass -p "ChangeMe" scp $ric:/root/*log* ./$out_dir/

echo "All tests complete"
kill $(jobs -p)

