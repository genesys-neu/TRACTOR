#!/bin/bash
# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id
## HOW TO RUN: ./setup_TRACTOR.sh genesys-<eNB #node> genesys-<UE #node> genesys-<RIC #node>

only_build=false
# ...do something interesting...
if [ "$only_build" = false ] ; then
  # for consistency, use the same radio.conf file
  sshpass -p "scope" scp ./utils/radio_tgen.conf $1:/root/radio_api/
  sshpass -p "scope" scp ./utils/radio_tgen.conf $2:/root/radio_api/
  
  # start the RF scenario + SCOPE on the gNB
  sshpass -p "scope" ssh $1 'colosseumcli rf start 10042 -c && cd /root/radio_api && python3 scope_start.py --config-file radio_tgen.conf'
  # start SCOPE on the UE
  sshpass -p "scope" ssh $2 'cd /root/radio_api && python3 scope_start.py --config-file radio_tgen.conf'
  # Start the Near-RT RIC
  sshpass -p "ChangeMe" ssh $3 'cd ~ && cd radio_code/colosseum-near-rt-ric/setup-scripts/ && ./setup-ric.sh col0'
fi

IPCOL0=`sshpass -p "ChangeMe" ssh $3 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo $IPCOL0
sshpass -p "scope" scp colosseum/radio_code/colosseum-scope-e2/src/du_app/csv_reader.* $1:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp colosseum/radio_code/colosseum-scope-e2/src/du_app/readLastMetrics.* $1:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp colosseum/radio_code/colosseum-scope-e2/src/du_app/bs_connector.* $1:/root/radio_code/colosseum-scope-e2/src/du_app/.
sshpass -p "scope" scp colosseum/radio_code/colosseum-scope-e2/build/odu/makefile $1:/root/radio_code/colosseum-scope-e2/build/odu/.
sshpass -p "scope" scp colosseum/radio_code/colosseum-scope-e2/run_odu.sh $1:/root/radio_code/colosseum-scope-e2/run_odu.sh
sshpass -p "scope" ssh $1 "cd /root/radio_code/colosseum-scope-e2/src/du_app/ && g++ readLastMetrics.cpp -o readLastMetrics.o"
sshpass -p "scope" ssh $1 "cd /root/radio_code/colosseum-scope-e2/ && sed -i 's/172.30.199.202/${IPCOL0}/' build_odu.sh && ./build_odu.sh clean" # && ./run_odu.sh

if [ "$only_build" = false ] ; then
  echo -e "*********************\n\tREADME\n*********************"
  echo -e "Now run the following commands on gNB ($1):\n\tsshpass -p \"scope\" ssh $1\n\tcd /root/radio_code/colosseum-scope-e2/\n\t./run_odu.sh"
  echo -e "Make sure that the e2 interface has been connected on the RIC side before proceeding:\n\tsshpass -p \"ChangeMe\" ssh $3 \"docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1\""
  echo -e "This command should return the gnb ID."
  echo -e "Then run:\n\tsh setup_TRACTOR_RIC.sh $1 $2 $3"
fi
