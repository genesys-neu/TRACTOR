# ARGS:
# 1 - eNB machine id
# 2 - RIC machine id


#!/bin/bash

IPeNB=`sshpass -p "scope" ssh $1 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo "eNB col0 IP $IPeNB"

IPRRIC=`sshpass -p "scope" ssh $2 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo "RIC col0 IP $IPRIC"

echo "*** Starting IPsec ***"
#Edit /etc/ipsec.secrets
sshpass -p "scope" ssh $1 "sed -i 's/10.207.208.180/${IPeNB}/;s/10.207.208.18/${IPRIC}/' /etc/ipsec.secrets"
sshpass -p "scope" ssh $2 "sed -i 's/10.207.208.180/${IPRIC}/;s/10.207.208.18/${IPeNB}/' /etc/ipsec.secrets"

#Edit /etc/ipsec.conf 
sshpass -p "scope" ssh $1 "sed -i 's/10.207.208.180/${IPeNB}/;s/10.207.208.18/${IPRIC}/' /etc/ipsec.conf"
sshpass -p "scope" ssh $2 "sed -i 's/10.207.208.180/${IPeNB}/;s/10.207.208.18/${IPRIC}/' /etc/ipsec.conf"

sshpass -p "scope" ssh $1 "sed -i 's/ike=aes256ccm128/#ike=aes256ccm128/;s/esp=aes256ccm128/#esp=aes256ccm128/' /etc/ipsec.conf"
sshpass -p "scope" ssh $2 "sed -i 's/ike=aes256ccm128/#ike=aes256ccm128/;s/esp=aes256ccm128/#esp=aes256ccm128/' /etc/ipsec.conf"

sshpass -p "scope" ssh $1 "sed -i 's/#ike=aes256gcm128/ike=aes256gcm128/;s/#esp=aes256gcm128/esp=aes256gcm128/' /etc/ipsec.conf"
sshpass -p "scope" ssh $2 "sed -i 's/#ike=aes256gcm128/ike=aes256gcm128/;s/#esp=aes256gcm128/esp=aes256gcm128/' /etc/ipsec.conf"

#Allow the kernel to access ipsec by Running: 
sshpass -p "scope" ssh $1 "apparmor_parser -R /etc/apparmor.d/usr.lib.ipsec.stroke" 
sshpass -p "scope" ssh $2 "apparmor_parser -R /etc/apparmor.d/usr.lib.ipsec.stroke" 

#Verify ipsec tunnel 
sshpass -p "scope" ssh $1 "ipsec restart" 
sshpass -p "scope" ssh $2 "ipsec restart" 
sleep 5
sshpass -p "scope" ssh $1 "ipsec status" 
sshpass -p "scope" ssh $2 "ipsec status" 

