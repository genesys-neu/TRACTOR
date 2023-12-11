# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id

#!/bin/bash

sshpass -p "ChangeMe" rsync -av -e ssh --exclude 'colosseum' --exclude '.git' --exclude 'logs/*UE/' --exclude 'utils/raw' --exclude 'raw' ../TRACTOR $3:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude 'colosseum' --exclude '.git' --exclude 'logs' --exclude 'utils/raw' --exclude 'model' ../TRACTOR $1:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude 'colosseum' --exclude '.git' --exclude 'logs' --exclude 'utils/raw' --exclude 'model' ../TRACTOR $2:/root/.

sshpass -p "ChangeMe" ssh $3 'docker cp /root/TRACTOR sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 mv /home/sample-xapp/TRACTOR/run_xapp.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp.sh'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 cp /home/sample-xapp/TRACTOR/mv_ts_files.sh /home/ && docker exec sample-xapp-24 chmod +x /home/mv_ts_files.sh'

echo -e "*********************\n\tREADME\n*********************"
echo -e "Now make sure the ODU is running on gNB ($1). If not, you can start it with:\n\tsshpass -p \"scope\" ssh $1\n\tcd /root/radio_code/colosseum-scope-e2/\n\t./run_odu.sh"
echo -e "Connect to the RIC and start the xapp:\n\tsshpass -p \"ChangeMe\" ssh $3 \n\tdocker exec -it sample-xapp-24 bash \n\t rm /home/*.log # remove previous logs\n\tcd /home/sample-xapp/ \n\t ./run_xapp.sh"
