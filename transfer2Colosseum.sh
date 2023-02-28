#!/bin/bash
sshpass -p "ChangeMe" ssh $1 'mkdir /root/logs'
sshpass -p "ChangeMe" scp ../../traffic_gen2/logs/cols_maxmin.pkl $1:/root/logs/.
sshpass -p "ChangeMe" scp ../../traffic_gen2/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl $1:/root/logs/.
sshpass -p "ChangeMe" ssh $1 'mkdir /root/model'
sshpass -p "ChangeMe" scp model/model_weights__slice8.pt $1:/root/model/.
sshpass -p "ChangeMe" ssh $1 'docker cp /root/logs/cols_maxmin.pkl sample-xapp-24:/home/sample-xapp/traffic_gen/logs/ && docker cp /root/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl sample-xapp-24:/home/sample-xapp/traffic_gen/logs/'
sshpass -p "ChangeMe" ssh $1 'docker cp /root/model/ sample-xapp-24:/home/sample-xapp/traffic_gen/.'
