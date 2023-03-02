This repository is organized as follows:

-logs contains the output of the data after it is played back in Colosseum. This folder has subfolders based on how and when I collected the data and what scenario was used to play it back in Colosseum. All files in the sub folders ending in _clean have been through initial data pre-processing.

-raw contains the captured real user traffic data 

-traffic_gen.py this is the script used to play back the raw data in colosseum.
To use: python traffic_gen.py -f <playback file name> [-e <specify if this is the base station>] [-i <Distant end IP address>] [-pb <eNB port>] [-pu <UE port>]

Always start the traffic generator on the eNB first using the -e (--eNB) flag.
Then start the traffic generator on the UE. Make sure to use the same file for both. 

It is recommended to use the -i (--ip) field for your implementation. The default IP address was configured for deployment in Colosseum using SCOPE. 
The ports do not need to be specified UNLESS you are implementing multiple instances of traffic on the same device. Then you must specify an unique port for each instance.

-python contains our ML models

## Setup TRACTOR on Colosseum
Requirements: 
  - Account for Colosseum
  - Local ssh config initialized as explained in their tutorial

Create a reservation on Colosseum involving 3 nodes and the following images (in this order):
- `groen-scope-w-e2` (gNB image) - root pwd: `scope`
- `groen-scope` (UE image) - root pwd: `scope`
- `groen-coloran-prebuilt` (RIC) - root pwd: `ChangeMe`

After initializing the nodes, call the following script:
```
sh setup_TRACTOR_gNB_UE.sh genesys-<gNB #node> genesys-<UE #node> genesys-<RIC #node>
```
This first script will initialize the gNB and UE LXC containers. After this script is complete, follow the instructions on the terminal in order to make sure the connection on E2 interface has been established and then proceed with initialization of the RIC:
```
sh setup_TRACTOR_RIC.sh genesys-<gNB #node> genesys-<UE #node> genesys-<RIC #node>
```
Once the RIC image has been deployed, launch the command to update the source code for the traffic classifier xApp:
```
sh transfer2Colosseum.sh genesys-<RIC #node>
```
Once the nodes have been initialized, you can connect to each of them and run the following commands (in this order):
  - On gNB
  ```
  cd /root/radio_code/colosseum-scope-e2 && ./run_odu.sh
  ```
  - On RIC
  ```
  docker exec -it sample-xapp-24 bash
  cd /home/sample-xapp/
  ./run_xapp.sh
  ```
  and wait until the app init is complete.
  - On UE
  ```
  iperf3 -c 172.16.0.1 -p 5204 -t <num seconds>
  ```
  where `<num seconds>` is the number of seconds to run for the traffic generator.
