#!/bin/bash
# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id

#echo "Waiting few seconds to make sure the E2 connection has been established.."
#sleep 15
GNBID=`sshpass -p "ChangeMe" ssh $3 "docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1"`
echo $GNBID
sshpass -p "ChangeMe" ssh $3 "cd /root/radio_code/colosseum-near-rt-ric/setup-scripts && ./setup-sample-xapp.sh ${GNBID}"
echo -e "Now you can transfer the code and data to the RIC\n\tsh transfer2Colosseum.sh $1 $2 $3"
