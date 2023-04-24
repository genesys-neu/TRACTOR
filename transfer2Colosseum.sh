# ARGS:
# 1 - gNB machine id
# 2 - UE machine id
# 3 - RIC machine id

#!/bin/bash
sshpass -p "ChangeMe" ssh $3 'mkdir /root/logs'
sshpass -p "ChangeMe" scp logs/cols_maxmin.pkl $3:/root/logs/.
#sshpass -p "ChangeMe" scp ../../traffic_gen2/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl $3:/root/logs/.
sshpass -p "ChangeMe" ssh $3 'mkdir /root/model'
sshpass -p "ChangeMe" scp model/model_weights__slice8.pt $3:/root/model/.

sshpass -p "ChangeMe" rsync -av -e ssh --exclude='colosseum' ../traffic_gen $3:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude='colosseum' ../traffic_gen $1:/root/.
sshpass -p "scope" rsync -av -e ssh --exclude='colosseum' ../traffic_gen $2:/root/.
#sshpass -p "ChangeMe" scp -r ../traffic_gen $3:/root/.
#sshpass -p "scope"  scp -r ../traffic_gen $1:/root/.
#sshpass -p "scope" scp -r ../traffic_gen $2:/root/.

sshpass -p "ChangeMe" ssh $3 'docker cp /root/traffic_gen sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $3 'docker cp /root/logs/cols_maxmin.pkl sample-xapp-24:/home/sample-xapp/traffic_gen/logs/ && docker cp /root/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl sample-xapp-24:/home/sample-xapp/traffic_gen/logs/'
sshpass -p "ChangeMe" ssh $3 'docker cp /root/model/ sample-xapp-24:/home/sample-xapp/traffic_gen/.'
#sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 ln -s /home/sample-xapp/xapp_control.py  /home/sample-xapp/traffic_gen/xapp_control.py'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 mv /home/sample-xapp/traffic_gen/run_xapp.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp.sh'
sshpass -p "ChangeMe" ssh $3 'docker exec sample-xapp-24 cp /home/sample-xapp/traffic_gen/mv_ts_files.sh /home/ && chmod +x /home/mv_ts_files.sh'

echo -e "*********************\n\tREADME\n*********************"
echo -e "Now make sure the ODU is running on gNB ($1). If not, you can start it with:\n\tsshpass -p \"scope\" ssh $1\n\tcd /root/radio_code/colosseum-scope-e2/\n\t./run_odu.sh"
echo -e "Connect to the RIC and start the xapp:\n\tsshpass -p \"ChangeMe\" ssh $3 \n\tdocker exec -it sample-xapp-24 bash \n\t rm /home/*.log # remove previous logs\n\tcd /home/sample-xapp/ \n\t ./run_xapp.sh"
