# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id

#!/bin/bash
sshpass -p "ChangeMe" ssh $3 'mkdir /root/logs'
sshpass -p "ChangeMe" scp logs/cols_maxmin.pkl $3:/root/logs/.
#sshpass -p "ChangeMe" scp ../../TRACTOR2/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl $3:/root/logs/.
sshpass -p "ChangeMe" ssh $3 'mkdir /root/model'
sshpass -p "ChangeMe" scp model/model_weights__slice8.pt $3:/root/model/.

sshpass -p "ChangeMe" rsync -av -e ssh --exclude='colosseum' ../TRACTOR $3:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude='colosseum' ../TRACTOR $1:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude='colosseum' ../TRACTOR $2:/root/.
#sshpass -p "ChangeMe" scp -r ../TRACTOR $3:/root/.
#sshpass -p "scope"  scp -r ../TRACTOR $1:/root/.
#sshpass -p "scope" scp -r ../TRACTOR $2:/root/.

sshpass -p "ChangeMe" ssh $3 'docker cp /root/TRACTOR sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $3 'docker cp /root/logs/cols_maxmin.pkl sample-xapp-24:/home/sample-xapp/TRACTOR/logs/ && docker cp /root/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl sample-xapp-24:/home/sample-xapp/TRACTOR/logs/'
sshpass -p "ChangeMe" ssh $3 'docker cp /root/model/ sample-xapp-24:/home/sample-xapp/TRACTOR/.'
#sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 ln -s /home/sample-xapp/xapp_control.py  /home/sample-xapp/TRACTOR/xapp_control.py'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 mv /home/sample-xapp/TRACTOR/run_xapp.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp.sh'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 cp /home/sample-xapp/TRACTOR/mv_ts_files.sh /home/ && docker exec sample-xapp-24 chmod +x /home/mv_ts_files.sh'

echo -e "*********************\n\tREADME\n*********************"
echo -e "Now make sure the ODU is running on gNB ($1). If not, you can start it with:\n\tsshpass -p \"scope\" ssh $1\n\tcd /root/radio_code/colosseum-scope-e2/\n\t./run_odu.sh"
echo -e "Connect to the RIC and start the xapp:\n\tsshpass -p \"ChangeMe\" ssh $3 \n\tdocker exec -it sample-xapp-24 bash \n\t rm /home/*.log # remove previous logs\n\tcd /home/sample-xapp/ \n\t ./run_xapp.sh"
