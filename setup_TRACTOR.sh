#!/bin/bash
## HOW TO RUN: ./setup_TRACTOR.sh genesys-<eNB #node> genesys-<UE #node> genesys-<RIC #node> 
# start the RF scenario + SCOPE on the gNB
sshpass -p "scope" ssh $1 'colosseumcli rf start 1017 -c && cd /root/radio_api && python3 scope_start.py --config-file radio_interactive.conf'
# start SCOPE on the UE
sshpass -p "scope" ssh $2 'cd /root/radio_api && python3 scope_start.py --config-file radio_interactive.conf'
# Start the Near-RT RIC
sshpass -p "ChangeMe" ssh $3 'cd ~ && cd radio_code/colosseum-near-rt-ric/setup-scripts/ && ./setup-ric.sh col0'
IPCOL0=`sshpass -p "ChangeMe" ssh $3 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo $IPCOL0
sshpass -p "scope" ssh $1 "cd /root/radio_code/colosseum-scope-e2/ && sed -i 's/172.30.199.202/${IPCOL0}/' build_odu.sh && ./build_odu.sh clean"
sshpass -p "ChangeMe" ssh $3 "cd /root/radio_code/colosseum-near-rt-ric/setup-scripts && ./setup-sample-xapp.sh gnb:311-048-01090801"
