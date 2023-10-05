import socket
import csv
import os
import time
import argparse
import sys
import random
import math



parser = argparse.ArgumentParser()
parser.add_argument("--eNB", help="specify this is the base station", action="store_true")
parser.add_argument("--ip", help="enter the distant end IP address", type=str)
parser.add_argument("-eNBp", "--eNBport", help="eNB port for this instance", type=int)
parser.add_argument("-UEp", "--UEport", help="UE port for this instance", type=int)
args = parser.parse_args()

# These are the default values
eNB_PORT = 5005
UE_PORT = 5115
data_size = 0

# add some arguments so we can specify a few options at run time
UE = not args.eNB

if args.eNBport:
    eNB_PORT = args.eNBport

if args.UEport:
    UE_PORT = args.UEport

if UE:
    local_port = UE_PORT
    distant_port = eNB_PORT
    Distant_IP = '172.16.0.1'
else:
    local_port = eNB_PORT
    distant_port = UE_PORT
    Distant_IP = '172.16.0.3'

if args.ip:    
    Distant_IP = args.ip

print("UDP target IP: %s" % Distant_IP)
print("UDP server port: %s" % eNB_PORT)
print("UDP client port: %s" % UE_PORT)

# sending UDP socket
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# receiving UDP socket
rec_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rec_sock.bind(('', local_port))

# iterating through the whole file
rowcount = 1000000

print("Number of entries in csv", rowcount)

if UE:  # The UE should always start
    print("[UE] I am the UE, I start communication")

    start_time = time.time()
    while rowcount > 0:
        data_size = int(math.ceil(random.gauss(mu=405, sigma=30)))
        Sdata = os.urandom(data_size)
        while time.time()-start_time < random.expovariate(lambd=1/0.000033):
            continue
        send_sock.sendto(Sdata, (Distant_IP, distant_port))
        if rowcount % 1000 == 0:
             print('[UE] Progress '+str(rowcount/1000000))
        rowcount -= 1
    
    print("[UE] test complete")

else:  # if we are the gNB
    print("[gNB] doesn't do anything")

