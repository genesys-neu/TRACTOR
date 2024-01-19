# Training TRACTOR models
All train-related scripts are in [python/](./) directory.
```
# from top repo directory
cd python/
```
## Generate sliced datasets
IMPORTANT: this code assumes that all CSV files have the same header/columns names and order.

First we use [ORAN_dataset.py](./ORAN_dataset.py) to generate the slice mapping from KPI logs:
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

python ORAN_dataset.py --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 --mode emuc --slicelen $l --data_type singleUE_raw --filemarker ${FILEMARKER} --cp_path ./${TRAINLOGDIR}/ --exp_name ICNC__slice${l}__${FILEMARKER} --already_gen --ds_pkl_paths ../logs/SingleUE/prev_experiments/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice${l}.pkl --normp_pkl ../logs/SingleUE/prev_experiments/cols_maxmin__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__singleUE_raw_noTs_prev_experiments.pkl
```
## Model Training 
The command [torch_train_ORAN.py](./torch_train_ORAN.py) it's used to train a given model using the dataset files just generated for this experiment. Possible models are:
- (default) Tansformer V1 (num. attention head `nhead = 1`, no CLS token is used)
- Transformer V2 (same as V1, but with CLS token implementation)
- CNN (see [TRACTOR](todo_add_reference) paper)

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
Note that `--relabel_train` applies relabeling of input data as explained on [MEGATRON](TODO_add_citation) paper. If you don't want to apply relabeling, simply run the same command without the `--relabel_train` flag.

Finally, generate the confusion matrix for the trained model using validation data:
```
python confusion_matrix.py --logdir ./${TRAINLOGDIR}/ICNC__slice${l}__${FILEMARKER}${SUFFIX}/
```
# Running model on pre-recorded KPI data
TODO..

# TODO: next steps
- Complete Visual Transformer support
- Add multiple attention heads to V1 and V2
- Re-test pipeline starting from CSV files (both Single and Multi UE)
- Finish description for running with pre-recorded Colosseum traces
- Add references to papers
