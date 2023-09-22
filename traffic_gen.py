import socket
import csv
import os
import time
import argparse
import sys



parser = argparse.ArgumentParser()
parser.add_argument("--eNB", help="specify this is the base station", action="store_true")
parser.add_argument("--ip", help="enter the distant end IP address", type=str)
parser.add_argument("-eNBp", "--eNBport", help="eNB port for this instance", type=int)
parser.add_argument("-UEp", "--UEport", help="UE port for this instance", type=int)
parser.add_argument("-f", "--file", help="full path to the csv file", type=str, required=True)
args = parser.parse_args()

# These are the default values
eNB_PORT = 5005
UE_PORT = 5115
data_size = 0

# add some arguments so we can specify a few options at run time
UE = not args.eNB
file_name = args.file

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
rowcount = 0
for row in open(file_name, 'r'):
    rowcount += 1
# printing the result
# discard first line
rowcount -= 1
print("Number of entries in csv", rowcount)

with open(file_name, 'r') as csvfile:

    datareader = csv.reader(csvfile)
    row1 = next(datareader)

    if UE:  # The UE should always start
        print("[UE] I am the UE, I start communication")
        row2 = next(datareader)
        # print(row2)
        if row2[3] != '172.30.1.1':
            print('[UE] eNB starts, send start message')
            send_sock.sendto(str.encode('Start'), (Distant_IP, distant_port))
            send_sock.sendto(str.encode('Start'), (Distant_IP, distant_port))
            # we also need to wait and listen for the first message
            while True:
                data, address = rec_sock.recvfrom(4096)
                if data:
                    start_time = time.time()
                    print("Starting experiment")
                    break

        else:
            # it is our turn to start
            data_size = int(row2[6])-70
            print('[UE] UE starts')
            start_time = time.time()
            Sdata = os.urandom(data_size)
            send_sock.sendto(Sdata, (Distant_IP, distant_port))

        for r_ix, row in enumerate(datareader):
            if r_ix % 100 == 0:
                print('[UE] Progress '+str(r_ix)+'/'+str(rowcount))
            if row[3] == '172.30.1.1':
                #print('[UE] It is our turn to send')
                data_size = int(row[6])-70
                Sdata = os.urandom(data_size)
                while time.time()-start_time < float(row[2]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))

            #else:
                # we should listen until we get data, or it is our turn to send again
                #print('[UE] listening')
                #while time.time() - start_time < float(row[2]):
                    #data, address = rec_sock.recvfrom(4096)
                    #if data:
                        #break

    else:  # if we are the eNB, we need to wait for a message from the UE before moving on
        print("[gNB] waiting for UE")

        while True:
            data, address = rec_sock.recvfrom(4096)
            if data:
                #print("[gNB] Starting experiment")
                start_time = time.time()
                break

        for r_ix, row in enumerate(datareader):
            if r_ix % 100 == 0:
                print('[gNB] Progress '+str(r_ix)+'/'+str(rowcount))
            if row[3] == '172.30.1.250':
                #print('[gNB] It is our turn to send')
                data_size = int(row[6])-70
                Sdata = os.urandom(data_size)
                while time.time()-start_time < float(row[2]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))

            #else:
                #print('[gNB] listening')
                # we should listen until we get data, or it is our turn to send again
                #while time.time() - start_time < float(row[2]):
                    #data, address = rec_sock.recvfrom(4096)
                    #if data:
                        #break

print('Test complete!')
