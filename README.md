# TRACTOR: Traffic Analysis and Classification Tool
We introduce TRACTOR: Traffic Analysis and Classification Tool for Open RAN. This classifier leverages the open interfaces of O-RAN to collect KPIs from the gNB. These KPIs, which do not contain any user-identifiable information, are used as inputs to ML classifiers in the near-RT RIC to classify user traffic.

This repository is organized as follows:

- logs contains the output of the data after it is played back in Colosseum. This folder has subfolders based on how and when I collected the data and what scenario was used to play it back in Colosseum. All files in the sub folders ending in _clean have been through initial data pre-processing.

- raw contains the captured real user traffic data 

- `traffic_gen.py` this is the script used to play back the raw data in colosseum.
To use: 
```
python traffic_gen.py -f <playback file name> [--eNB <specify if this is the base station>] [--ip <Distant end IP address>] [-eNBp <eNB port>] [-UEp <UE port>]
```

Always start the traffic generator on the eNB first using the --eNB flag.
Then start the traffic generator on the UE. Make sure to use the same file for both. 

It is recommended to use the --ip field for your implementation. The default IP address was configured for deployment in Colosseum using SCOPE. 
The ports do not need to be specified UNLESS you are implementing multiple instances of traffic on the same device. Then you must specify an unique port for each instance.

- python contains our ML models

- utils contains additional utilities

## Setup TRACTOR on Colosseum
Requirements: 
  - Account for Colosseum
  - Local ssh config initialized as explained in their tutorial

### Reservations
Create a reservation on Colosseum involving 3 nodes and the following images (in this order):
- `groen-scope-w-e2` (gNB image) - root pwd: `scope`
- `groen-scope` (UE image) - root pwd: `scope`
- `groen-coloran-prebuilt` (RIC) - root pwd: `ChangeMe`

### Nodes initialization
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
sh transfer2Colosseum.sh genesys-<gNB #node> genesys-<UE #node> genesys-<RIC #node>
```
### Start gNB and RIC containers
Once the nodes have been initialized, you can connect to each of them and run the following commands (in this order):
- On gNB (*NOTE: This only needs to be done if run_odu.sh is not already running from previous steps.*)
```
cd /root/radio_code/colosseum-scope-e2 && ./run_odu.sh
```
- On RIC, run the following command specifying, in the following order:
  1. Path of model to be loaded for live traffic classification (`.pt`);
  2. Path of normalization parameters used during training (`.pkl`);
  3. Model type, chosen from `['CNN', 'Tv1', 'Tv2']` based on the type of model loaded.

```
docker exec -it sample-xapp-24 bash
cd /home/sample-xapp/
./run_xapp.sh <model_path> <norm_param_path> <model_type>
```
and wait until the app init is complete. NOTE: the paths passed as arguments have to reference files placed in docker container running the xApp.

### Replay 5G traffic traces
To automatically run traffic generation, you can use :
```
run_traffic_gNB_UE.sh genesys-gNB genesys-UE genesys-RIC
```
There are additional traffic generation utilities in the /utils directory.

To *manually re-play* traffic traces found in the /raw directory, after everything else is instantiated you can first run on gNB
```
python traffic_gen.py -f <playback file name> --eNB 
```
and then on UE
```
python traffic_gen.py -f <playback file name>
```
to start the predefined packet communication between gNB and UE on Colosseum.

### Generate random traffic from UE to the gNB
To saturate the RF link with random traffic, on the gNB node run:
```
iperf3 -s -p 5204 -D
```
And on UE node run:
```
iperf3 -c 172.16.0.1 -p 5204 -t <num seconds>
```
where `<num seconds>` is the number of seconds to run for the traffic generator and `172.16.0.1` corresponds to the IP of the gNB in Colosseum.

### Other Utilities 
Additional utilities are found in the /utils directory. These utilities include utilizing multiple UEs, generating interference, generating malicious traffic, and enabling IPsec on the E2 interface.

