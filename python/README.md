# Training TRACTOR models
All train-related scripts are in [python/](python/) directory.
## Generate sliced datasets
First we use [ORAN_dataset.py](python/ORAN_dataset.py) to generate the slice mapping from KPI logs:
```
$ python ORAN_dataset.py --help
usage: ORAN_dataset.py [-h] [--trials [TRIALS ...]] [--trials_multi [TRIALS_MULTI ...]] [--filemarker FILEMARKER] [--slicelen SLICELEN] [--ds_path DS_PATH]
                       [--check_zeros] [--mode {emu,emuc,co}] [--data_type {singleUE_clean,singleUE_raw,multiUE} [{singleUE_clean,singleUE_raw,multiUE} ...]]
                       [--drop_colnames [DROP_COLNAMES ...]] [--already_gen]

optional arguments:
  -h, --help            show this help message and exit
  --trials [TRIALS ...]
                        Trials in SingleUE KPI data folder eg. "Trail1 Trail2 Trail3"
  --trials_multi [TRIALS_MULTI ...]
                        Trials in MultiUE data folder eg. "Trail1 Trail2 Trail3"
  --filemarker FILEMARKER
                        Suffix added to the file as marker
  --slicelen SLICELEN   Specify the slices lengths while generating the dataset.
  --ds_path DS_PATH     Specify path where dataset files are stored
  --check_zeros         Assign ctrl label to slices which all their rows contain >10 zeros
  --mode {emu,emuc,co}  This argument specifies which class to use when generating the dataset: 1) "emu" means all classes except CTRL; 2) "emuc" include CTRL
                        class; 3) "co" is specific to CTRL traffic and will generate a separate class for every other type of traffic.
  --data_type {singleUE_clean,singleUE_raw,multiUE} [{singleUE_clean,singleUE_raw,multiUE} ...]
                        This argument specifies the type of KPI traces to read into the dataset.
  --drop_colnames [DROP_COLNAMES ...]
                        Remove specified column names from data frame when loaded from .csv files.s
  --already_gen         [DEBUG] Use this flag for pre-generated dataset(s) that are only needed to compute new statistics.
```

TODO: explain how this work.

IMPORTANT: this code assumes that all CSV files have the same header/columns names and order.

### Single UE (raw) traces
This is to replicate the initial results obtained only with Single UE traces. 
```
python ORAN_dataset.py --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 --mode emuc --data_type singleUE_raw --slicelen 16 --filemarker noTimestamp --drop_colnames Timestamp
```
Now train with basic Transformer (no positional autoencoder, 1 head, no CLS token)
```
python torch_train_ORAN.py --ds_file SingleUE/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice16_singleUE_raw_noTimestamp.pkl --isNorm --ds_path ../logs --cp_path ./train_logs/ --norm_param_path SingleUE/cols_maxmin__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice16_singleUE_raw_noTimestamp.pkl --exp_name no_timestamp --relabel_train --transformer v1
```
