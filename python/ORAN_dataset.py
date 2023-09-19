import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
import pickle
import argparse
from glob import glob
import sys
from visualize_inout import classmap

def load_csv_traces(trials, data_path, data_type="clean", norm_params=None):
    mode = 'emuc'
    isControlClass = True if 'c' in mode else False
    trials_traces = []
    for trial in trials:
        print('Generating dataset ', trial)
        if data_type == "clean":
            ctrl_data, embb_data, mmtc_data, urll_data = load_csv_dataset__clean(data_path, trial, isControlClass)

            # stack together all data from all traffic class
            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                datasets = [embb_data, mmtc_data, urll_data, ctrl_data]
            else:
                datasets = [embb_data, mmtc_data, urll_data]

            for ix, ds in enumerate(datasets):
                # let's first remove undesired columns from the dataframe (these features are not relevant for traffic classification)
                columns_drop = ['Timestamp', 'tx_errors downlink (%)']  # ['Timestamp', 'tx_errors downlink (%)', 'dl_cqi']
                ds.drop(columns_drop, axis=1, inplace=True)
                # normalize values
                for ix, c in enumerate(ds.columns):
                    #print('Normalizing Col.', c, '-- Max', norm_params[ix]['max'], ', Min', norm_params[ix]['min'])
                    ds[c] = ds[c].map(lambda x: (x - norm_params[ix]['min']) / (norm_params[ix]['max'] - norm_params[ix]['min']))

            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                trials_traces.append({'embb': embb_data, 'mmtc': mmtc_data, 'urll': urll_data, 'ctrl': ctrl_data})
            else:
                trials_traces.append({'embb': embb_data, 'mmtc': mmtc_data, 'urll': urll_data})

        elif data_type == "multi":
            ds_tree = load_csv_dataset__multi(data_path, trial, isControlClass)
            #TODO

    return trials_traces

def check_slices(data, index, check_zeros=False):
    labels = np.ones((data.shape[0],), dtype=np.int32)*index
    if not check_zeros:
        return labels
    for i in range(data.shape[0]):
        sl = data[i]
        zeros = (sl == 0).astype(int).sum(axis=1)
        if (zeros > 10).all():
            labels[i] = 3 # control if all KPIs rows have > 10 zeros
    return labels
        


def gen_slice_dataset(trials, data_path, slice_len=4, train_valid_ratio=0.8, mode='emuc', data_type="clean", check_zeros=False):

    isControlClass = True if 'c' in mode else False
    allowed_types = ["clean", "raw", "multi"]
    allowed_modes = ["emuc", "emu", "oc"]
    assert (data_type in allowed_types), "[gen_slice_dataset] ERROR: data_type must be chosen from:"+str(allowed_types)
    assert (mode in allowed_modes), "[gen_slice_dataset] ERROR: class configuration \""+str(mode)+"\" is not supported. Aborting."

    trials_in = []
    trials_lbl = []
    for trial in trials:
        print('Generating dataset ', trial)
        if data_type == "clean":
            ctrl_data, embb_data, mmtc_data, urll_data = load_csv_dataset__clean(data_path, trial, isControlClass)

            # stack together all data from all traffic class
            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                datasets = [embb_data, mmtc_data, urll_data, ctrl_data]
            else:
                datasets = [embb_data, mmtc_data, urll_data]

            for ix, ds in enumerate(datasets):
                # let's first remove undesired columns from the dataframe (these features are not relevant for traffic classification)
                columns_drop = ['Timestamp', 'tx_errors downlink (%)'] # ['Timestamp', 'tx_errors downlink (%)', 'dl_cqi']
                ds.drop(columns_drop, axis=1, inplace=True)
                new_ds = slice_dataset(ds, slice_len)

                print("Generating ORAN traffic KPI dataset")
                # create labels based on dataset generation mode
                if mode == 'emu' or mode == 'emuc':
                    # 4 traffic classes problem
                    if ix == 0:
                        print("\teMBB class")
                        embb_data = new_ds
                        embb_labels = check_slices(embb_data, ix, check_zeros) # labels are numbers (i.e. no 1 hot encoded)
                    elif ix == 1:
                        print("\tMMTc class")
                        mmtc_data = new_ds
                        mmtc_labels = check_slices(mmtc_data, ix, check_zeros)
                    elif ix == 2:
                        print("\tURLLc class")
                        urll_data = new_ds
                        urll_labels = check_slices(urll_data, ix, check_zeros)
                    elif ix == 3:
                        print("\tControl / CTRL class")
                        ctrl_data = new_ds
                        ctrl_labels = np.ones((ctrl_data.shape[0],), dtype=np.int32)*ix
                elif mode == 'co':
                    # binary traffic labeling (i.e. yes/no active traffic
                    if ix != 3:
                        print("Active traffic class")
                        if ix == 0:
                            embb_data = new_ds
                            embb_labels = np.zeros((embb_data.shape[0],), dtype=np.int32) # labels are numbers (i.e. no 1 hot encoded)
                        elif ix == 1:
                            mmtc_data = new_ds
                            mmtc_labels = np.zeros((mmtc_data.shape[0],), dtype=np.int32)
                        elif ix == 2:
                            urll_data = new_ds
                            urll_labels = np.zeros((urll_data.shape[0],), dtype=np.int32)
                    else:
                        print("\tControl / CTRL class")
                        ctrl_data = new_ds
                        ctrl_labels = np.ones((ctrl_data.shape[0],), dtype=np.int32)
                else:
                    print("[gen_slice_dataset] ERROR: class configuration \"", mode, "\" is not supported. Aborting.")
                    sys.exit(1)

            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                all_input = np.concatenate((embb_data, mmtc_data, urll_data, ctrl_data), axis=0)
                all_labels = np.concatenate((embb_labels, mmtc_labels, urll_labels, ctrl_labels), axis=0)
            else:
                all_input = np.concatenate((embb_data, mmtc_data, urll_data), axis=0)
                all_labels = np.concatenate((embb_labels, mmtc_labels, urll_labels), axis=0)

        elif data_type == "multi":
            all_input = None
            all_labels = None
            ds_tree = load_csv_dataset__multi(data_path, trial, isControlClass)

            for multi_conf in ds_tree.keys():
                for u in ds_tree[multi_conf].keys():
                    # process each user trace individually
                    ds = ds_tree[multi_conf][u]['kpis']
                    # slicing
                    columns_drop = []   # TODO determine what columns to drop
                    if len(columns_drop) > 0:
                        ds.drop(columns_drop, axis=1, inplace=True)
                    new_ds = slice_dataset(ds, slice_len)

                    lbl = ds_tree[multi_conf][u]["label"]
                    if lbl == "embb" or lbl == "mmtc" or lbl == "urll":
                        class_ix = classmap[lbl]
                        labels = check_slices(new_ds, class_ix, check_zeros)
                    elif lbl == "ctrl":
                        class_ix = 4
                        labels = np.ones((new_ds.shape[0],), dtype=np.int32) * class_ix

                    if all_input is None:
                        all_input = np.array(new_ds)
                    else:
                        all_input = np.concatenate((all_input, new_ds), axis=0)

                    if all_labels is None:
                        all_labels = labels
                    else:
                        all_labels = np.concatenate((all_labels, labels))

        # TODO --> elif data_type == "raw":

        trials_in.append(all_input)
        trials_lbl.append(all_labels)

    trials_in = np.concatenate(trials_in, axis=0).astype(np.float32)
    trials_lbl = np.concatenate(trials_lbl, axis=0).astype(int)

    # also create a normalized version of the features (each feature is normalized independently)
    trials_in_norm = trials_in.copy()

    columns_maxmin = {}
    for c in range(trials_in_norm.shape[2]):
        col_max = trials_in_norm[:,:,c].max()
        col_min = trials_in_norm[:,:,c].min()
        print('Normalizing Col.', c, '-- Max', col_max, ', Min', col_min)
        trials_in_norm[:, :, c] = (trials_in_norm[:, :, c] - col_min) / (col_max - col_min)
        # store max/min values for later normalization
        columns_maxmin[c] = {'max': col_max, 'min': col_min}

    # generate (shuffled) train and test data
    samp_ixs = list(range(trials_in.shape[0]))
    np.random.shuffle(samp_ixs)
    # shuffle the dataset samples and labels in the same order
    trials_in = trials_in[samp_ixs, :, :]
    trials_in_norm = trials_in_norm[samp_ixs, :, :]
    trials_lbl = trials_lbl[samp_ixs]

    if mode == 'emuc':
        nclasses = 4
    elif mode == 'emu':
        nclasses = 3
    elif mode == 'co':
        nclasses = 2

    for c in range(nclasses):
        nsamps_c = np.where(trials_lbl == c)[0].shape[0]
        print('Class', c, '=', nsamps_c, 'samples')

    n_samps = trials_in.shape[0]
    n_train = int(n_samps * train_valid_ratio)
    n_valid = n_samps - n_train

    trials_ds = {
        'train': {
            'samples': {
                'no_norm': torch.Tensor(trials_in[0:n_train]),
                'norm': torch.Tensor(trials_in_norm[0:n_train])
            },
            'labels': torch.Tensor(trials_lbl[0:n_train]).type(torch.LongTensor)
        },
        'valid': {
            'samples': {
                'no_norm': torch.Tensor(trials_in[n_train:n_train+n_valid]),
                'norm': torch.Tensor(trials_in_norm[n_train:n_train+n_valid])
            },
            'labels': torch.Tensor(trials_lbl[n_train:n_train+n_valid]).type(torch.LongTensor)
        }
    }

    return trials_ds, columns_maxmin


def slice_dataset(ds, slice_len):
    new_ds = []
    for i in tqdm(range(ds.shape[0]), desc='Slicing..'):
        if i + slice_len < ds.shape[0]:
            new_ds.append(
                ds[i:i + slice_len]  # slice
            )
    new_ds = np.array(new_ds)
    return new_ds


def load_csv_dataset__clean(data_path, trial, isControlClass=True):
    # for each traffic type, let's load csv info using pandas
    embb_files = glob(os.path.join(data_path, os.path.join(trial, "embb_*clean.csv")))
    embb_data = pd.concat([pd.read_csv(f, sep=",") for f in embb_files])
    mmtc_files = glob(os.path.join(data_path, os.path.join(trial, "mmtc_*clean.csv")))
    mmtc_data = pd.concat([pd.read_csv(f, sep=",") for f in mmtc_files])
    urll_files = glob(os.path.join(data_path, os.path.join(trial, "urll*_*clean.csv")))
    urll_data = pd.concat([pd.read_csv(f, sep=",") for f in urll_files])
    if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
        ctrl_data = pd.read_csv(os.path.join(data_path, os.path.join(trial, "null_clean.csv")), sep=",")
    else:
        ctrl_data = None
    # drop specific columns
    if 'ul_rssi' in mmtc_data.columns:
        mmtc_data = mmtc_data.drop(['ul_rssi'], axis=1)
    return ctrl_data, embb_data, mmtc_data, urll_data

def load_csv_dataset__multi(data_path, trial, isControlClass=False):
    ds_tree = {}
    # find all multi-ue configurations
    trial_path = os.path.join(data_path, trial)
    multiUE_list = glob(os.path.join(trial_path, 'multi*'))
    for m in multiUE_list:
        multi_key = os.path.basename(m)
        label_file = glob(os.path.join(m, '*.txt'))
        assert(len(label_file) == 1)
        # obtain the labels for each user
        ds_tree[multi_key] = {}
        with open(label_file[0], 'r') as f:
            for line in f:
                split_line = line.split()
                trace_filename = split_line[2]
                ueID = split_line[4]
                label = ''
                if "mmtc" in trace_filename.lower():
                    label = "mmtc"
                elif ("urll" in trace_filename.lower()) or ("urllc" in trace_filename.lower()):
                    label = "urll"
                elif "embb" in trace_filename.lower():
                    label = "embb"
                assert(label != '')

                ue_csv = os.path.join(m, ueID+"_metrics.csv")
                ue_kpis = pd.read_csv(ue_csv, sep=",")
                ue_kpis = ue_kpis.dropna(how='all', axis='columns')  # remove empty columns
                ds_tree[multi_key][str(ueID)] = {"trace": trace_filename, "label": label, "kpis": ue_kpis }

    return ds_tree



class ORANTracesDataset(Dataset):
    def __init__(self, dataset_pkl, key, normalize=True, path='/home/mauro/Research/ORAN/TRACTOR/logs/', transform=None, target_transform=None):
        dataset = pickle.load(open(os.path.join(path, dataset_pkl), 'rb'))
        if normalize:
            norm_key = 'norm'
        else:
            norm_key = 'no_norm'
        self.obs_input, self.obs_labels = dataset[key]['samples'][norm_key], dataset[key]['labels']
        self.transform = transform
        self.target_transform = target_transform

    def info(self):
        ds_info = {
            'numfeats': self.obs_input.shape[2],
            'slice_len': self.obs_input.shape[1],
            'numsamps': self.obs_input.shape[0],
            'nclasses': len(np.unique(np.array(self.obs_labels)))
        }
        return ds_info

    def __len__(self):
        return len(self.obs_input)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.obs_labels.iloc[idx, 0])
        #image = read_image(img_path)       # we load at run time instead that in __init__
        #label = self.obs_labels.iloc[idx, 1]
        obs = self.obs_input[idx, :, :]
        label = self.obs_labels[idx]

        if self.transform:
            obs = self.transform(obs)
        if self.target_transform:
            label = self.target_transform(label)
        return obs, label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, default=['Trial1', 'Trial2', 'Trial3'], nargs='+', help="Trials in the data folder eg. \"Trail1 Trail2 Trail3\"")
    parser.add_argument("--filemarker", default='', help="Suffix added to the file as marker")
    parser.add_argument("--slicelen", default=4, type=int, help="Specify the slices lengths while generating the dataset.")
    parser.add_argument("--ds_path", default="/home/mauro/Research/ORAN/repo/TRACTOR/logs/", help="Specify path where dataset files are stored")
    parser.add_argument("--check_zeros", action="store_true", default=False, help="Assign ctrl label to slices which all their rows contain >10 zeros")
    parser.add_argument("--mode", default='emuc', choices=['emu', 'emuc', 'co'],
                        help='This argument specifies which class to use when generating the dataset: '
                             '1) "emu" means all classes except CTRL; '
                             '2) "emuc" include CTRL class; '
                             '3) "co" is specific to CTRL traffic and will generate a separate class for every other type of traffic.')
    parser.add_argument("--data_type", default="clean", choices=["clean", "raw", "multi"], help="This argument specifies the type of KPI traces to load.")
    args, _ = parser.parse_known_args()

    path = args.ds_path
    trials = args.trials
    dataset, cols_maxmin = gen_slice_dataset(trials, slice_len=args.slicelen, data_path=path, mode=args.mode, data_type=args.data_type, check_zeros=args.check_zeros)

    file_suffix = ''
    for t in trials:
        file_suffix += t
        if t != trials[-1]:
            file_suffix += '_'

    file_suffix += '__slice'+str(args.slicelen)
    file_suffix += '_'+args.filemarker if args.filemarker else ''
    zeros_suffix = '_ctrlcorrected_' if args.check_zeros else ''
    pickle.dump(dataset, open(os.path.join(path, 'dataset__' + args.mode + '__' +file_suffix + zeros_suffix + '.pkl'), 'wb'))

    # save separately maxmin normalization parameters for each column/feature
    norm_param_path = os.path.join(path,'cols_maxmin__'+file_suffix+'.pkl')
    yes_choice = ['yes', 'y']
    save_norm_param = True
    if os.path.isfile(norm_param_path):
        user_input = input("File "+norm_param_path+" exists already. Overwrite? [y/n]")
        if not(user_input.lower() in yes_choice):
            save_norm_param = False

    if save_norm_param:
        pickle.dump(cols_maxmin, open(norm_param_path, 'wb'))
