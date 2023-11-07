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

def load_csv_traces(trials, data_path, data_type="singleUE_clean", norm_params=None, isRaw=False):
    mode = 'emuc'
    isControlClass = True if 'c' in mode else False
    trials_traces = []
    for trial in trials:
        print('Generating dataset ', trial)
        if data_type == "singleUE_clean":
            ctrl_traces, embb_traces, mmtc_traces, urll_traces = load_csv_dataset__single(data_path, trial, isControlClass, isRaw)
            # stack together all data from all traffic class
            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                datasets = [embb_traces, mmtc_traces, urll_traces, ctrl_traces]
            else:
                datasets = [embb_traces, mmtc_traces, urll_traces]
            # preprocessing
            for ix, ds in enumerate(datasets):
                for trace in ds:
                    # let's first remove undesired columns from the dataframe (these features are not relevant for traffic classification)
                    columns_drop = ['Timestamp', 'tx_errors downlink (%)']  # ['Timestamp', 'tx_errors downlink (%)', 'dl_cqi']
                    trace.drop(columns_drop, axis=1, inplace=True)
                    # normalize values
                    for ix, c in enumerate(trace.columns):
                        # print('Normalizing Col.', c, '-- Max', norm_params[ix]['max'], ', Min', norm_params[ix]['min'])
                        trace[c] = trace[c].map(lambda x: (x - norm_params[ix]['min']) / (norm_params[ix]['max'] - norm_params[ix]['min']))

            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                trials_traces.append({'embb': embb_traces, 'mmtc': mmtc_traces, 'urll': urll_traces, 'ctrl': ctrl_traces})
            else:
                trials_traces.append({'embb': embb_traces, 'mmtc': mmtc_traces, 'urll': urll_traces})

        elif data_type == "multiUE":
            ds_tree = load_csv_dataset__multi(data_path, trial, isControlClass)
            for multi_conf in ds_tree.keys():
                for u in ds_tree[multi_conf].keys():
                    # process each user trace individually
                    ds = ds_tree[multi_conf][u]['kpis']

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


data_type_choice = ["singleUE_clean", "singleUE_raw", "multiUE"]


def gen_slice_dataset(trials, data_path, slice_len=4, train_valid_ratio=0.8,
                      mode='emuc', data_type="singleUE_clean",
                      check_zeros=False, drop_colnames=[]):

    isControlClass = True if 'c' in mode else False
    allowed_types = data_type_choice
    allowed_modes = ["emuc", "emu", "oc"]
    assert (data_type in allowed_types), "[gen_slice_dataset] ERROR: data_type must be chosen from:"+str(allowed_types)
    assert (mode in allowed_modes), "[gen_slice_dataset] ERROR: class configuration \""+str(mode)+"\" is not supported. Aborting."

    trials_in = []
    trials_lbl = []

    if data_type == "singleUE_clean" or data_type == "singleUE_raw":
        data_path = os.path.join(data_path, 'SingleUE')
    else:
        data_path = os.path.join(data_path, 'Multi-UE')

    cols_names = None

    for trial in trials:
        print('Generating dataset ', trial)

        if data_type == "singleUE_clean" or data_type == "singleUE_raw":
            isRaw = "_raw" in data_type
            ctrl_logs, embb_logs, mmtc_logs, urll_logs = load_csv_dataset__single(data_path, trial, isControlClass, isRaw)
            # stack together all data from all traffic class
            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                datasets = [embb_logs, mmtc_logs, urll_logs, ctrl_logs]
            else:
                datasets = [embb_logs, mmtc_logs, urll_logs]

            # assuming all the files have same headers..
            if cols_names is None:
                cols_names = embb_logs[0].columns.values

            embb_traces = []
            embb_labels = []
            mmtc_traces = []
            mmtc_labels = []
            urll_traces = []
            urll_labels = []
            ctrl_traces = []
            ctrl_labels = []

            for ix, ds in enumerate(datasets):

                for trace in ds:    # analyze each trace for each traffic type

                    # let's first remove undesired columns from the dataframe (these features are not relevant for traffic classification)
                    if not isRaw:
                        columns_drop = ['Timestamp', 'tx_errors downlink (%)'] # ['Timestamp', 'tx_errors downlink (%)', 'dl_cqi']
                    else:
                        columns_drop = drop_colnames
                    trace.drop(columns_drop, axis=1, inplace=True)

                    new_trace = slice_dataset(trace, slice_len)

                    print("Generating ORAN traffic KPI dataset")
                    # create labels based on dataset generation mode
                    if mode == 'emu' or mode == 'emuc':
                        # 4 traffic classes problem
                        if ix == 0:
                            print("\teMBB class")
                            embb_traces.append(new_trace)
                            embb_labels.append(check_slices(new_trace, ix, check_zeros)) # labels are numbers (i.e. no 1 hot encoded)
                        elif ix == 1:
                            print("\tMMTc class")
                            mmtc_traces.append(new_trace)
                            mmtc_labels.append(check_slices(new_trace, ix, check_zeros))
                        elif ix == 2:
                            print("\tURLLc class")
                            urll_traces.append(new_trace)
                            urll_labels.append(check_slices(new_trace, ix, check_zeros))
                        elif ix == 3:
                            print("\tControl / CTRL class")
                            ctrl_traces.append(new_trace)
                            ctrl_labels.append(np.ones((new_trace.shape[0],), dtype=np.int32)*ix)
                    elif mode == 'co':
                        # binary traffic labeling (i.e. yes/no active traffic
                        if ix != 3:
                            print("Active traffic class")
                            if ix == 0:
                                embb_traces.append(new_trace)
                                embb_labels.append( np.zeros((new_trace.shape[0],), dtype=np.int32) ) # labels are numbers (i.e. no 1 hot encoded)
                            elif ix == 1:
                                mmtc_traces.append(new_trace)
                                mmtc_labels.append( np.zeros((new_trace.shape[0],), dtype=np.int32) )
                            elif ix == 2:
                                urll_traces.append(new_trace)
                                urll_labels.append( np.zeros((urll_traces.shape[0],), dtype=np.int32) )
                        else:
                            print("\tControl / CTRL class")
                            ctrl_traces.append(new_trace)
                            ctrl_labels.append( np.ones((ctrl_traces.shape[0],), dtype=np.int32) )
                    else:
                        print("[gen_slice_dataset] ERROR: class configuration \"", mode, "\" is not supported. Aborting.")
                        sys.exit(1)

            # consolidate all the samples generated for each traffic type
            if len(embb_traces) > 0 and len(embb_labels) > 0:
                embb_traces = np.concatenate(embb_traces)
                embb_labels = np.concatenate(embb_labels)
            if len(mmtc_traces) > 0 and len(mmtc_labels) > 0:
                mmtc_traces = np.concatenate(mmtc_traces)
                mmtc_labels = np.concatenate(mmtc_labels)
            if len(urll_traces) > 0 and len(urll_traces) > 0:
                urll_traces = np.concatenate(urll_traces)
                urll_labels = np.concatenate(urll_labels)
            if len(ctrl_traces) > 0 and len(ctrl_labels) > 0:
                ctrl_traces = np.concatenate(ctrl_traces)
                ctrl_labels = np.concatenate(ctrl_labels)

            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):
                all_input = np.concatenate((embb_traces, mmtc_traces, urll_traces, ctrl_traces), axis=0)
                all_labels = np.concatenate((embb_labels, mmtc_labels, urll_labels, ctrl_labels), axis=0)
            else:
                all_input = np.concatenate((embb_traces, mmtc_traces, urll_traces), axis=0)
                all_labels = np.concatenate((embb_labels, mmtc_labels, urll_labels), axis=0)

        elif data_type == "multiUE":

            all_input = None
            all_labels = None
            ds_tree = load_csv_dataset__multi(data_path, trial, isControlClass)

            for multi_conf in ds_tree.keys():
                for u in ds_tree[multi_conf].keys():
                    print("Generating slices: ", multi_conf, u)
                    # process each user trace individually
                    ds = ds_tree[multi_conf][u]['kpis']
                    if cols_names is None:
                        cols_names = ds.columns.values
                    # slicing
                    columns_drop = drop_colnames
                    if len(columns_drop) > 0:
                        ds.drop(columns_drop, axis=1, inplace=True)
                    new_trace = slice_dataset(ds, slice_len)

                    lbl = ds_tree[multi_conf][u]["label"]
                    if lbl == "embb" or lbl == "mmtc" or lbl == "urll":
                        class_ix = classmap[lbl]
                        labels = check_slices(new_trace, class_ix, check_zeros)
                    elif lbl == "ctrl":
                        class_ix = 4
                        labels = np.ones((new_trace.shape[0],), dtype=np.int32) * class_ix

                    if all_input is None:
                        all_input = np.array(new_trace)
                    else:
                        all_input = np.concatenate((all_input, new_trace), axis=0)

                    if all_labels is None:
                        all_labels = labels
                    else:
                        all_labels = np.concatenate((all_labels, labels))

        trials_in.append(all_input)
        trials_lbl.append(all_labels)

    # if we are using Timestamp, we need to convert them to relative Timestamps before converting everything to float32
    # because the Timestamp integer requires 64 bits to be represented correctly, otherwise it gets truncated
    # TODO would it be better todo this relative to the whole trace rather than on a slice basis?
    if cols_names[0] == 'Timestamp':
        for trial in trials_in:
            trial = np.stack([relative_timestamp(x) for x in trial])

    trials_in = np.concatenate(trials_in, axis=0).astype(np.float32)
    trials_lbl = np.concatenate(trials_lbl, axis=0).astype(int)

    # also create a normalized version of the features (each feature is normalized independently)
    trials_in_norm = trials_in.copy()

    columns_maxmin = {}
    for c in range(trials_in_norm.shape[2]):
        if cols_names[c] != 'Timestamp':
            col_max = trials_in_norm[:, :, c].max()
            col_min = trials_in_norm[:, :, c].min()
            if not (col_max == col_min):
                print('Normalizing Col.', cols_names[c], '[', c, '] -- Max', col_max, ', Min', col_min)
                trials_in_norm[:, :, c] = (trials_in_norm[:, :, c] - col_min) / (col_max - col_min)
            else:
                trials_in_norm[:, :, c] = 0
            # store max/min values for later normalization
            columns_maxmin[c] = {'max': col_max, 'min': col_min, 'name': cols_names[c]}

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


def load_csv_dataset__single(data_path, trial, isControlClass=True, isRaw=False):
    # for each traffic type, let's load csv info using pandas

    if isRaw:
        embb_csv = "Raw/embb*.csv"
        urrl_csv = "Raw/urll*.csv"
        mmtc_csv = "Raw/mmtc*.csv"
        clean_csv = "Raw/null*.csv"
    else:
        embb_csv = "embb_*clean.csv"
        urrl_csv = "urll*_*clean.csv"
        mmtc_csv = "mmtc_*clean.csv"
        clean_csv = "null_clean.csv"

    embb_files = glob(os.path.join(data_path, os.path.join(trial, embb_csv)))
    embb_traces = [pd.read_csv(f, sep=",").dropna(how='all', axis='columns') for f in embb_files] # also remove completely blank columns, if present

    mmtc_files = glob(os.path.join(data_path, os.path.join(trial, mmtc_csv)))
    mmtc_traces = [pd.read_csv(f, sep=",").dropna(how='all', axis='columns') for f in mmtc_files]

    urll_files = glob(os.path.join(data_path, os.path.join(trial, urrl_csv)))
    urll_traces = [pd.read_csv(f, sep=",").dropna(how='all', axis='columns') for f in urll_files]

    clean_files = glob(os.path.join(data_path, os.path.join(trial, clean_csv)))
    if isControlClass and len(clean_files) > 0:
        ctrl_traces = [pd.read_csv(f, sep=",").dropna(how='all', axis='columns') for f in clean_files]
    else:
        ctrl_traces = None


    # drop specific columns
    # from chat with Josh:
    #   [15/11/22 17:44] Mauro Belgiovine
    #       Joshua Groen is it possible that in Trial2, mmtc data have an additional column? I see there is ul_rssi that I haven't seen in any other f
    #   [15/11/22 17:51] Joshua Groen
    #       ul_rssi does not work; it should be deleted. It is possible I missed something
    if not isRaw:
        # TODO consider if doing this for every other case
        for tr_ix, traces in enumerate(mmtc_traces):
            if 'ul_rssi' in traces.columns:
                mmtc_traces[tr_ix] = traces.drop(['ul_rssi'], axis=1)


    return ctrl_traces, embb_traces, mmtc_traces, urll_traces

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


def relative_timestamp(x):
    first_ts = x[0,0] # get first value of first column (Timestamp)
    x[:, 0] -=  first_ts
    return x

class ORANTracesDataset(Dataset):
    def __init__(self, dataset_pkls, key, normalize=True, path='../logs/', sanitize=True, relabel_CTRL=False, relabel_norm_threshold=2., transform=None, target_transform=None):
        self.obs_input, self.obs_labels = None, None
        for p in dataset_pkls:
            dataset = pickle.load(open(os.path.join(path, p), 'rb'))
            if normalize:
                norm_key = 'norm'
            else:
                norm_key = 'no_norm'
            if (self.obs_input is None) and (self.obs_labels is None):
                self.obs_input, self.obs_labels = dataset[key]['samples'][norm_key], dataset[key]['labels']
            else:
                self.obs_input = torch.cat((self.obs_input, dataset[key]['samples'][norm_key]), dim=0)
                self.obs_labels = torch.cat((self.obs_labels, dataset[key]['labels']), dim=0)

        # sanitize columns: remove columns with no variation of data
        if sanitize:
            obs_std = np.std(self.obs_input.numpy(), axis=(0, 1))
            features_to_remove = np.where(obs_std == 0)[0]
            all_feats = torch.arange(0,self.obs_input.shape[-1])
            indexes_to_keep = [i for i in range(len(all_feats)) if i not in features_to_remove]
            self.obs_input = self.obs_input[:, :, indexes_to_keep]

        self.sanitized = sanitize

        self.ctrl_label = 3
        self.relabel_norm_threshold = relabel_norm_threshold
        self.relabeled = False
        if relabel_CTRL:
            self.relabel_ctrl_samples()

        self.transform = transform
        self.target_transform = target_transform

    def relabel_ctrl_samples(self):
        ixs_ctrl = self.obs_labels == self.ctrl_label
        all_ctrl = self.obs_input[ixs_ctrl]
        include_ixs = [1] + [x for x in range(3, self.obs_input.shape[-1])]  # exclude column 0 (Timestamp) and 2 (IMSI)
        mean_ctrl_sample = torch.mean(all_ctrl[:, :, include_ixs], dim=0)
        # compute euclidean distance between samples of other classes and mean ctrl sample
        obs_excludecols = self.obs_input[:, :, include_ixs]
        norm = np.linalg.norm(obs_excludecols - mean_ctrl_sample, axis=(1, 2))
        # here we get the indexes for all samples that have a norm (i.e. euclidean distance) less than a given
        # threshold, which should correspond to CTRL (i.e. silent) traffic portions. Note that the lower the threshold,
        # the more conservative is the relabeling. This threshold is computed based on the distribution of euclidean
        # distances computed between the mean CTRL sample (assuming they look very similar)
        # and every samples of every other class
        possible_ctrl_ixs = norm < self.relabel_norm_threshold
        for ix, isPossibleCTRL in enumerate(possible_ctrl_ixs):
            if isPossibleCTRL and self.obs_labels[ix] != self.ctrl_label:
                self.obs_labels[ix] = self.ctrl_label
        #possible_ctrl_labels = self.obs_labels[possible_ctrl_ixs].numpy()

        self.relabeled = True


    def info(self):
        unique_labels, unique_count = np.unique(np.array(self.obs_labels), return_counts=True)
        ds_info = {
            'numfeats': self.obs_input.shape[2],
            'slice_len': self.obs_input.shape[1],
            'numsamps': self.obs_input.shape[0],
            'nclasses': len(unique_labels),
            'samples_per_class': unique_count
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
    parser.add_argument("--trials", default=['Trial1', 'Trial2', 'Trial3', 'Trial4', 'Trial5', 'Trial6'], nargs='*', help="Trials in SingleUE KPI data folder eg. \"Trail1 Trail2 Trail3\"")
    parser.add_argument("--trials_multi", default=['Trial2', 'Trial3'], nargs='*', help="Trials in MultiUE data folder eg. \"Trail1 Trail2 Trail3\"")
    parser.add_argument("--filemarker", default='', help="Suffix added to the file as marker")
    parser.add_argument("--slicelen", default=4, type=int, help="Specify the slices lengths while generating the dataset.")
    parser.add_argument("--ds_path", default="../logs/", help="Specify path where dataset files are stored")
    parser.add_argument("--check_zeros", action="store_true", default=False, help="Assign ctrl label to slices which all their rows contain >10 zeros")
    parser.add_argument("--mode", default='emuc', choices=['emu', 'emuc', 'co'],
                        help='This argument specifies which class to use when generating the dataset: '
                             '1) "emu" means all classes except CTRL; '
                             '2) "emuc" include CTRL class; '
                             '3) "co" is specific to CTRL traffic and will generate a separate class for every other type of traffic.')
    parser.add_argument("--data_type", default="singleUE_clean", nargs='+', choices=data_type_choice, help="This argument specifies the type of KPI traces to read into the dataset.")
    parser.add_argument("--drop_colnames", default=[], nargs='*', help="Remove specified column names from data frame when loaded from .csv files.s")
    args, _ = parser.parse_known_args()

    from visualize_inout import classmap

    path = args.ds_path
    trials = args.trials
    trials_multi = args.trials_multi
    for data_type in args.data_type:
        trials = trials_multi if "multi" in data_type else trials
        dataset, cols_maxmin = gen_slice_dataset(trials, slice_len=args.slicelen, data_path=path,
                                                 mode=args.mode, data_type=data_type, check_zeros=args.check_zeros,
                                                 drop_colnames=args.drop_colnames)

        file_suffix = ''
        for t in trials:
            file_suffix += t
            if t != trials[-1]:
                file_suffix += '_'

        file_suffix += '__slice' + str(args.slicelen) +'_' + data_type
        file_suffix += '_'+args.filemarker if args.filemarker else ''
        zeros_suffix = '_ctrlcorrected_' if args.check_zeros else ''
        pickle_ds_path = os.path.join(path, "Multi-UE") if "multi" in data_type else os.path.join(path, "SingleUE")
        pickle.dump(dataset, open(os.path.join(pickle_ds_path, 'dataset__' + args.mode + '__' +file_suffix + zeros_suffix + '.pkl'), 'wb'))

        # save separately maxmin normalization parameters for each column/feature
        norm_param_path = os.path.join(pickle_ds_path,'cols_maxmin__'+file_suffix+'.pkl')
        yes_choice = ['yes', 'y']
        save_norm_param = True
        if os.path.isfile(norm_param_path):
            user_input = input("File "+norm_param_path+" exists already. Overwrite? [y/n]")
            if not(user_input.lower() in yes_choice):
                save_norm_param = False

        if save_norm_param:
            pickle.dump(cols_maxmin, open(norm_param_path, 'wb'))
