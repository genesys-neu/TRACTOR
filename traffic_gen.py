import socket
import csv
import os
import time
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--eNB", help="specify if this is the base station", action="store_true")
parser.add_argument("-i", "--ip", help="enter the distant end IP address", type=str)
parser.add_argument("-pb", "--port_eNB", help="eNB port for this instance", type=int)
parser.add_argument("-pu", "--port_UE", help="UE port for this instance", type=int)
parser.add_argument("-f", "--file", help="full path to the csv file", type=str, required=True)
args = parser.parse_args()

# These are the default values
eNB_PORT = 5005
UE_PORT = 5115
data_size = 0

# add some arguments so we can specify a few options at run time
UE = not args.eNB
file_name = args.file

if args.port_eNB:
    eNB_PORT = args.port_eNB

if args.port_UE:
    UE_PORT = args.port_UE

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

with open(file_name, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    row1 = next(datareader)

    if UE:  # The UE should always start
        print("I am the UE, I start communication")
        row2 = next(datareader)
        # print(row2)
        if row2[3] != '172.30.1.1':
            print('eNB starts, send start message')
            send_sock.sendto(str.encode('Start'), (Distant_IP, distant_port))
            start_time = time.time()
            print('starting experiment')
            print('listening for %s' % row2[1])
            # we also need to wait and listen for the first message
            while time.time() - start_time < float(row2[2]):
                data, address = rec_sock.recvfrom(4096)
                if data:
                    break
        else:
            # it is our turn to start
            data_size = int(row2[6])-70
            print('UE starts')
            start_time = time.time()
            Sdata = os.urandom(data_size)
            send_sock.sendto(Sdata, (Distant_IP, distant_port))

        for row in datareader:
            if row[3] == '172.30.1.1':
                data_size = int(row[6])-70
                Sdata = os.urandom(data_size)
                while time.time()-start_time < float(row[2]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))
                print('Sending %s' % row[1])

            else:
                # we should listen until we get data, or it is our turn to send again
                print('listening for %s' % row[1])
                while time.time() - start_time < float(row[2]):
                    data, address = rec_sock.recvfrom(4096)
                    if data:
                        break

    else:  # if we are the eNB, we need to wait for a message from the UE before moving on
        print("waiting for UE")

        while True:
            data, address = rec_sock.recvfrom(4096)
            if data:
                start_time = time.time()
                print("Starting experiment")
                break

        for row in datareader:
            if row[3] == '172.30.1.250':
                data_size = int(row[6])-70
                Sdata = os.urandom(data_size)
                while time.time()-start_time < float(row[2]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))
                print('Sending %s' % row[1])

            else:
                print('listening for %s' % row[1])
                # we should listen until we get data, or it is our turn to send again
                while time.time() - start_time < float(row[2]):
                    data, address = rec_sock.recvfrom(4096)
                    if data:
                        break

print('Test completed with the following parameters:')
print("UDP target IP: %s" % Distant_IP)
print("UDP server port: %s" % eNB_PORT)
print("UDP client port: %s" % UE_PORT)
print("File used: %s" % file_name)
