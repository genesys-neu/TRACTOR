This folder is organized as follows:

-logs contains the output of the data after it is played back in Colosseum. This folder has subfolders based on
how and when I collected the data. All files in the sub folders ending in _clean have been through initial data 
pre-processing.

-raw contains the raw captured data - I only move the raw .csv into this folder after it is played back in Colosseum

-traffic_gen.py this is the script used to play back the raw data in colosseum.
To use: python traffic_gen.py -file <playback file name> [--eNB] [-i <Distant end IP address>] [-pb <eNB port>] [-pu <UE port>]
Always start the traffic generator on the eNB first using the --eNB flag.
Then start the traffic generator on the UE. Make sure to use the same file for both! It is recommended to use the -i (--ip) field for your implementation.
Default IP address was configured for deployment in Colosseum using SCOPE. 
The ports do not need to be specified UNLESS you are implementing multiple instances of traffic on the same device. Then you must specify an unique port for each instance.

-any other .csv in this folder will be raw data that has NOT YET been played back in Colosseum
