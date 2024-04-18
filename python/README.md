# Training and testing TRACTOR models
All train-related scripts are in [`python/`](./) directory.
```
# from top repo directory
cd python/
```
## Generate sliced datasets
IMPORTANT: this code assumes that all CSV files have the same header/columns names and order.

First we use [`ORAN_dataset.py`](./ORAN_dataset.py) to generate the slice mapping from KPI logs:
```
$ python ORAN_dataset.py --help
usage: ORAN_dataset.py [-h] [--trials [TRIALS ...]] [--trials_multi [TRIALS_MULTI ...]] [--filemarker FILEMARKER] [--slicelen SLICELEN] [--ds_path DS_PATH] [--check_zeros] [--mode {emu,emuc,co}]
                       [--data_type {singleUE_clean,singleUE_raw,multiUE} [{singleUE_clean,singleUE_raw,multiUE} ...]] [--drop_colnames [DROP_COLNAMES ...]] [--already_gen] [--exp_name EXP_NAME] [--cp_path CP_PATH]
                       [--ds_pkl_paths DS_PKL_PATHS] [--normp_pkl NORMP_PKL] [--exclude_cols EXCLUDE_COLS [EXCLUDE_COLS ...]]

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
  --mode {emu,emuc,co}  This argument specifies which class to use when generating the dataset: 1) "emu" means all classes except CTRL; 2) "emuc" include CTRL class; 3) "co" is specific to CTRL traffic and will generate a separate
                        class for every other type of traffic.
  --data_type {singleUE_clean,singleUE_raw,multiUE} [{singleUE_clean,singleUE_raw,multiUE} ...]
                        This argument specifies the type of KPI traces to read into the dataset.
  --drop_colnames [DROP_COLNAMES ...]
                        Remove specified column names from data frame when loaded from .csv files.s
  --already_gen         [DEBUG] Use this flag for pre-generated dataset(s) that are only needed to compute new statistics.
  --exp_name EXP_NAME   Name of this experiment
  --cp_path CP_PATH     Path to save/load checkpoint and training/dataset logs
  --ds_pkl_paths DS_PKL_PATHS
                        (--already-gen) specify origin pkl file.
  --normp_pkl NORMP_PKL
                        (--already-gen) specify origin pkl file.
  --exclude_cols EXCLUDE_COLS [EXCLUDE_COLS ...]
                        (--already-gen) specify origin pkl file.
```

### Single UE (raw) traces (pre-generated dataset)
It's recommended to train with pre-generated dataset, for sake of reproducibility. To do that, first, download the [pre-generated](https://drive.google.com/drive/folders/1HXShC1yaSPyoGaOZjO1KqO9POARzBICq?usp=drive_link) dataset and copy it in `../logs/SingleUE/`; then, run the following command:

This is to replicate the initial results obtained only with Single UE traces. In order to generate the necessary dataset files, run the following command using `--already_gen` option: 
```
FILEMARKER=prevexp_globalnorm
SUFFIX=_meanthr
TRAINLOGDIR=train_log6

l=16 # slice length

python ORAN_dataset.py --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 --mode emuc --slicelen $l --data_type singleUE_raw --filemarker ${FILEMARKER} --cp_path ./${TRAINLOGDIR}/ --exp_name ICNC__slice${l}__${FILEMARKER}${SUFFIX} --already_gen --ds_pkl_paths ../logs/SingleUE/prev_experiments/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice${l}.pkl --normp_pkl ../logs/SingleUE/prev_experiments/cols_maxmin__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__singleUE_raw_noTs_prev_experiments.pkl
```
### Single UE (raw) traces (newly collected datset, aka `Trial7`)
[`Trial7`](../logs/SingleUE/Trial7/Raw) consists of all the traces in Trials 1-6 re-processed through SCOPE/Colosseum in a uniform configuration (i.e. single UE allocated within a single traffic slice assigned with all available Resource Blocks (RBs)). This was done to provide a more uniform set of input features and prevent the classifier from associating specific testbed configurations to certain traffic types. Note that this folder also contains the test traces previously used for [TRACTOR](https://arxiv.org/abs/2312.07896) and [MEGATRON](http://www.conf-icnc.org/2024/papers/p1054-belgiovine.pdf) papers (`*_04_10.csv`) that in this case are re-used for training/validation. 

In order to pre-process the dataset using this data, run the following command:
```
FILEMARKER=newcollect
SUFFIX=_meanthr
TRAINLOGDIR=train_log10

l=16 # slice length
python ORAN_dataset.py --trials Trial7 --mode emuc --slicelen $l --data_type singleUE_raw --drop_colnames Timestamp num_ues IMSI RNTI slicing_enabled slice_id slice_prb scheduling_policy --cp_path ./${TRAINLOGDIR}/ --filemarker trial7__slice${l}__${FILEMARKER}${SUFFIX}
```

## Model Training 
The command [`torch_train_ORAN.py`](./torch_train_ORAN.py) it's used to train a given model using the dataset files just generated for this experiment. Possible models are:
- (default) Tansformer V1 (num. attention head `nhead = 1`, no CLS token is used)
- Transformer V2 (same as V1, but with CLS token implementation)
- CNN (see [TRACTOR](https://arxiv.org/abs/2312.07896) paper)
- Visual Transformer (ViT) with pre-defined configuration (see `megatron_ViT` class in [`ORAN_models.py`](./ORAN_models.py))

```
usage: torch_train_ORAN.py [-h] --ds_file DS_FILE [DS_FILE ...] [--ds_path DS_PATH] [--isNorm] [--test {val,traces}] [--relabel_test] [--relabel_train] [--cp_path CP_PATH] [--exp_name EXP_NAME] [--norm_param_path NORM_PARAM_PATH]
                           [--transformer {v1,v2,ViT}] [--pos_enc] [--patience PATIENCE] [--lrmax LRMAX] [--lrmin LRMIN] [--lrpatience LRPATIENCE] [--useRay] [--info_verbose] [--address ADDRESS] [--num-workers NUM_WORKERS]
                           [--use-gpu]

optional arguments:
  -h, --help            show this help message and exit
  --ds_file DS_FILE [DS_FILE ...]
                        Name of dataset pickle file containing training data and labels.
  --ds_path DS_PATH     Specify path where dataset files are stored
  --isNorm              Specify to load the normalized dataset.
  --test {val,traces}   Testing the model
  --relabel_test        Perform ctrl label correction during testing time
  --relabel_train       Perform ctrl label correction during training time
  --cp_path CP_PATH     Path to save/load checkpoint and training logs
  --exp_name EXP_NAME   Name of this experiment
  --norm_param_path NORM_PARAM_PATH
                        Normalization parameters path.
  --transformer {v1,v2,ViT}
                        Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)
  --pos_enc             Use positional encoder (only applied to transformer arch)
  --patience PATIENCE   Num of epochs to wait before interrupting training with early stopping
  --lrmax LRMAX         Initial learning rate
  --lrmin LRMIN         Final learning rate after scheduling
  --lrpatience LRPATIENCE
                        Patience before triggering learning rate decrease
  --useRay              Run training using Ray
  --info_verbose        Print/plot some info about dataset visualization.
  --address ADDRESS     [Deprecated] the address to use for Ray
  --num-workers NUM_WORKERS, -n NUM_WORKERS
                        [Deprecated] Sets number of workers for training.
  --use-gpu             [Deprecated] Enables GPU training
```
### Training Transformer V1 on Single UE dataset
Now train with basic Transformer (no positional autoencoder, 1 head, no CLS token)
```
    python torch_train_ORAN.py --ds_file ../logs/SingleUE/prev_experiments/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice${l}__globalnorm.pkl --isNorm --ds_path ../logs --cp_path ./${TRAINLOGDIR}/ --norm_param_path ../logs/global__cols_maxmin__${FILEMARKER}_slice${l}.pkl --exp_name ICNC__slice${l}__${FILEMARKER}${SUFFIX} --transformer v1 --relabel_train
```
Note that `--relabel_train` applies relabeling of input data as explained on [MEGATRON](http://www.conf-icnc.org/2024/papers/p1054-belgiovine.pdf) paper. If you don't want to apply relabeling, simply run the same command without the `--relabel_train` flag.

Finally, generate the confusion matrix for the trained model using validation data:
```
python confusion_matrix.py --logdir ./${TRAINLOGDIR}/ICNC__slice${l}__${FILEMARKER}${SUFFIX}/
```
# Running model on pre-recorded KPI data
If Colosseum traces for test data are available, it is possible to run [`visual_xapp_inference.py`](./visual_xapp_inference.py) in order to obtain the accuracy of prediction of a certain trace. Note that CSV traces should contain a single traffic type in it and have the keyword relative to the traffic type at the beginning of the file name (i.e. `embb_*.csv`, `mmtc_*.csv` or `urllc_*.`) in order to compare classifier's results using the correct traffic type. To deal with *idle* portion of the traffic, the same CTRL/idle mean template and threshold mechanism used for filtering the training dataset (see description of `--relabel_train` above) is used to detect whether the classifier output is validated or not through the proposed filtering heuristic: if classifier's output is equal to class `3` (i.e. CTRL/idle) and the sample passes the heuristic test, we validate the classifier decision and consider the output as correctly classified as `idle` traffic. 

```
usage: visual_xapp_inference.py [-h] --trace_path TRACE_PATH [--mode {pre-comp,inference,inference_offline}]
                                [--slicelen {4,8,16,32,64}] [--model_path MODEL_PATH] [--norm_param_path NORM_PARAM_PATH]
                                [--model_type {CNN,Tv1,Tv1_old,Tv2,ViT}] [--Nclasses NCLASSES] [--dir_postfix DIR_POSTFIX]
                                [--CTRLcheck] [--chZeros]

optional arguments:
  -h, --help            show this help message and exit
  --trace_path TRACE_PATH
                        Path containing the classifier output files for re-played traffic traces
  --mode {pre-comp,inference,inference_offline}
                        Specify the type of file format we are trying to read.
  --slicelen {4,8,16,32,64}
                        Specify the slicelen to determine the classifier to load
  --model_path MODEL_PATH
                        Path to TRACTOR model to load.
  --norm_param_path NORM_PARAM_PATH
                        Normalization parameters path.
  --model_type {CNN,Tv1,Tv1_old,Tv2,ViT}
                        Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)
  --Nclasses NCLASSES   Used to initialize the model
  --dir_postfix DIR_POSTFIX
                        This is appended to the name of the output folder for images and text
  --CTRLcheck           At test time (inference), it will compare the sample with CTRL template to determine if its a correct CTRL
                        sample
  --chZeros             [Deprecated] At test time, don't count the occurrences of ctrl class
```

It is possible to loop this function over all test traces in a folder with the following shell command:
```
for i in `ls <path-to-test-folder>/*.csv`; do python visual_xapp_inference.py --trace_path $i --slicelen $l --model_path <path-to-model-.pt> --norm_param_path <path-to-norm-param-.pkl> --mode inference_offline --CTRLcheck --model_type <model-type> --dir_postfix <str-appended-to-output-dir>; done
```

# TODO: next steps
- [x] Complete Visual Transformer support.
- [ ] Add multiple attention heads to V1 and V2
- [x] Re-test pipeline starting from CSV files (both Single and Multi UE)
- [x] Finish description for running with pre-recorded Colosseum traces
- [ ] Add example output for offline inference output
- [x] Add references to papers
