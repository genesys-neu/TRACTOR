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
from scipy.stats import norm as normal
import sys

def load_csv_traces(trials, data_path, data_type="singleUE_clean", norm_params=None):
    mode = 'emuc'
    isControlClass = True if 'c' in mode else False
    trials_traces = []
    for trial in trials:
        print('Generating dataset ', trial)
        if data_type == "singleUE_clean":
            isRaw = False
            ctrl_traces, embb_traces, mmtc_traces, urll_traces = load_csv_dataset__single(data_path, trial, isControlClass, isRaw)
            # stack together all data from all traffic class
            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))): # TODO: remove the .csv check
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

            if isControlClass and os.path.exists(os.path.join(data_path, os.path.join(trial, "null_clean.csv"))):  # TODO: remove the .csv check
                trials_traces.append({'embb': embb_traces, 'mmtc': mmtc_traces, 'urll': urll_traces, 'ctrl': ctrl_traces})
            else:
                trials_traces.append({'embb': embb_traces, 'mmtc': mmtc_traces, 'urll': urll_traces})

        elif data_type == "multiUE":
            # TODO finish coding this part (needed at test time)
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
    excluded_indexes_sUE = None
    all_cols_names_sUE = None
    excluded_indexes_mUE = None
    all_cols_names_mUE = None

    for trial in trials:
        print('Generating dataset ', trial)

        if data_type == "singleUE_clean" or data_type == "singleUE_raw":
            trial_samples, cols_n, excluded_indexes_sUE, all_cols_names_sUE = get_trace_singleUE(data_path,
                                                       data_type,
                                                       drop_colnames,
                                                       isControlClass,
                                                       mode,
                                                       slice_len,
                                                       trial)

            if cols_names is None:  # NOTE: assume Trials have all the same columns, we set only once
                cols_names = cols_n

            if isControlClass and not(len(trial_samples['ctrl']['traces']) == 0):
                all_input = np.concatenate((trial_samples['embb']['traces'],
                                            trial_samples['mmtc']['traces'],
                                            trial_samples['urll']['traces'],
                                            trial_samples['ctrl']['traces']), axis=0)
                all_labels = np.concatenate((trial_samples['embb']['labels'],
                                             trial_samples['mmtc']['labels'],
                                             trial_samples['urll']['labels'],
                                             trial_samples['ctrl']['labels']), axis=0)
            else:
                all_input = np.concatenate((trial_samples['embb']['traces'],
                                            trial_samples['mmtc']['traces'],
                                            trial_samples['urll']['traces'],), axis=0)
                all_labels = np.concatenate((trial_samples['embb']['labels'],
                                             trial_samples['mmtc']['labels'],
                                             trial_samples['urll']['labels'], ), axis=0)

        elif data_type == "multiUE":

            trial_samples, cols_n, all_input, all_labels, excluded_indexes_mUE, all_cols_names_mUE = get_trace_multiUE(data_path,
                                                                             drop_colnames,
                                                                             isControlClass,
                                                                             mode,
                                                                             slice_len,
                                                                             trial)

            if cols_names is None:  # NOTE: assume Trials have all the same columns, we set only once
                cols_names = cols_n


        trials_in.append(all_input)
        trials_lbl.append(all_labels)

    if (not (excluded_indexes_sUE is None ) and not (excluded_indexes_mUE is None)) and (not(all_cols_names_sUE is None) and not (all_cols_names_mUE is None)):
        assert np.all(excluded_indexes_sUE == excluded_indexes_mUE) and np.all(all_cols_names_sUE == all_cols_names_mUE), "Check these are matching for all datasets processed"

    excluded_indexes = excluded_indexes_sUE # since are the same after previous check
    all_cols_names = all_cols_names_sUE
    if (excluded_indexes is None) and (all_cols_names is None):
        excluded_indexes = excluded_indexes_mUE # to address multi UE only dataset case
        all_cols_names = all_cols_names_mUE

    # if we are using Timestamp, we need to convert them to relative Timestamps before converting everything to float32
    # because the Timestamp integer requires 64 bits to be represented correctly, otherwise it gets truncated
    # TODO: would it be better todo this relative to the whole trace rather than on a slice basis?
    if cols_names[0] == 'Timestamp':
        for trial in trials_in:
            trial = np.stack([relative_timestamp(x) for x in trial])    # only needed if we want to use the experimental Positional Encoder (not working now)

    trials_in = np.concatenate(trials_in, axis=0).astype(np.float32)
    trials_lbl = np.concatenate(trials_lbl, axis=0).astype(int)

    #columns_maxmin, trials_in_norm = normalize_KPIs(cols_names, trials_in)

    columns_maxmin = extract_feats_stats(cols_names, trials_in)
    trials_in_norm = normalize_KPIs(columns_maxmin, trials_in)

    columns_maxmin['info'] = {'raw_cols_names': all_cols_names, 'exclude_cols_ix': excluded_indexes}

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


def get_trace_multiUE(data_path, drop_colnames, isControlClass, mode,
                      slice_len, trial, use_multi_conf='all', check_zeros=False):
    """
    This function is used to generate (sample, label) pairs from MultiUE traces. Note that each Trial contains
    mutliple configurations of transmitting UEs (e.g. "multi4" means 4 UEs associated to the same base station)

    :param data_path:
    :param drop_colnames:
    :param isControlClass:
    :param mode:
    :param slice_len:
    :param trial:
    :param use_multi_conf:
    :param check_zeros:
    :return:
    """
    all_input = None
    all_labels = None
    all_cols_names = None
    cols_names = None
    excluded_indexes = None

    ds_tree = load_csv_dataset__multi(data_path, trial, isControlClass)
    sliced_ds = {}
    for multi_conf in ds_tree.keys():
        if use_multi_conf == 'all' or multi_conf == use_multi_conf:
            for u in ds_tree[multi_conf].keys():
                print("Generating slices: ", multi_conf, u)
                # process each user trace individually
                ds = ds_tree[multi_conf][u]['kpis']
                columns_drop = drop_colnames

                # NOTE: assuming all the files have same headers and same columns are dropped
                if excluded_indexes is None:  # done only once for all datasets
                    excluded_indexes = [ds.columns.get_loc(c) for c in columns_drop]
                if all_cols_names is None:  # only once
                    all_cols_names = ds.columns.values  # before drop

                if len(columns_drop) > 0:
                    ds.drop(columns_drop, axis=1, inplace=True)

                # obtain column names after drop
                # (assuming all the files have same columns and same columns dropped)
                if cols_names is None:  # done only once
                    cols_names = ds.columns.values
                # slicing
                # TODO adjust this and single UE function to return traces without slicing
                new_trace = slice_dataset(ds, slice_len)

                lbl = ds_tree[multi_conf][u]["label"]
                if mode == 'emu' or mode == 'emuc':
                    if lbl == "embb" or lbl == "mmtc" or lbl == "urll":
                        class_ix = classmap[lbl]
                        labels = check_slices(new_trace, class_ix, check_zeros)
                    elif lbl == "ctrl":
                        class_ix = 4
                        labels = np.ones((new_trace.shape[0],), dtype=np.int32) * class_ix
                elif mode == 'co':
                    if lbl == "embb" or lbl == "mmtc" or lbl == "urll":
                        labels = np.zeros((new_trace.shape[0],), dtype=np.int32)
                    elif lbl == "ctrl":
                        labels = np.ones((new_trace.shape[0],), dtype=np.int32)

                sliced_ds[lbl] = {'traces': new_trace, 'labels': labels}

                if all_input is None:
                    all_input = np.array(new_trace)
                else:
                    all_input = np.concatenate((all_input, new_trace), axis=0)

                if all_labels is None:
                    all_labels = labels
                else:
                    all_labels = np.concatenate((all_labels, labels))

    return sliced_ds, cols_names, all_input, all_labels, excluded_indexes, all_cols_names


def get_trace_singleUE(data_path, data_type, drop_colnames, isControlClass, mode, slice_len,
                       trial, check_zeros=False):
    isRaw = "_raw" in data_type
    ctrl_logs, embb_logs, mmtc_logs, urll_logs = load_csv_dataset__single(data_path, trial, isControlClass, isRaw)
    # stack together all data from all traffic class
    if isControlClass and len(ctrl_logs) > 0:
        datasets = [embb_logs, mmtc_logs, urll_logs, ctrl_logs]
    else:
        datasets = [embb_logs, mmtc_logs, urll_logs]

    embb_traces = []
    embb_labels = []
    mmtc_traces = []
    mmtc_labels = []
    urll_traces = []
    urll_labels = []
    ctrl_traces = []
    ctrl_labels = []

    all_cols_names = None
    cols_names = None
    excluded_indexes = None

    for ix, ds in enumerate(datasets):

        for trace in ds:  # analyze each trace for each traffic type

            # let's first remove undesired columns from the dataframe (these features are not relevant for traffic classification)
            if not isRaw:
                columns_drop = ['Timestamp',
                                'tx_errors downlink (%)']  # ['Timestamp', 'tx_errors downlink (%)', 'dl_cqi']
            else:
                columns_drop = drop_colnames

            # NOTE: assuming all the files have same headers and same columns are dropped
            if excluded_indexes is None:    # done only once for all datasets
                excluded_indexes = [trace.columns.get_loc(c) for c in columns_drop]
            if all_cols_names is None:  # only once
                all_cols_names = trace.columns.values   # before drop

            trace.drop(columns_drop, axis=1, inplace=True)

            # NOTE: assuming all the files have same headers and same columns are dropped
            if cols_names is None:   # done only once for all datasets
                cols_names = trace.columns.values   # after drop

            # slicing
            new_trace = slice_dataset(trace, slice_len)

            print("Generating ORAN traffic KPI dataset")
            # create labels based on dataset generation mode
            if mode == 'emu' or mode == 'emuc':
                # 4 traffic classes problem
                if ix == 0:
                    print("\teMBB class")
                    embb_traces.append(new_trace)
                    embb_labels.append(
                        check_slices(new_trace, ix, check_zeros))  # labels are numbers (i.e. no 1 hot encoded)
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
                    ctrl_labels.append(np.ones((new_trace.shape[0],), dtype=np.int32) * ix)
            elif mode == 'co':
                # binary traffic labeling (i.e. yes/no active traffic
                if ix != 3:
                    print("Active traffic class")
                    if ix == 0:
                        embb_traces.append(new_trace)
                        embb_labels.append(np.zeros((new_trace.shape[0],),
                                                    dtype=np.int32))  # labels are numbers (i.e. no 1 hot encoded)
                    elif ix == 1:
                        mmtc_traces.append(new_trace)
                        mmtc_labels.append(np.zeros((new_trace.shape[0],), dtype=np.int32))
                    elif ix == 2:
                        urll_traces.append(new_trace)
                        urll_labels.append(np.zeros((urll_traces.shape[0],), dtype=np.int32))
                else:
                    print("\tControl / CTRL class")
                    ctrl_traces.append(new_trace)
                    ctrl_labels.append(np.ones((ctrl_traces.shape[0],), dtype=np.int32))
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

    return {
        'ctrl': {'traces': ctrl_traces, 'labels': ctrl_labels},
        'embb': {'traces': embb_traces, 'labels': embb_labels},
        'mmtc': {'traces': mmtc_traces, 'labels': mmtc_labels},
        'urll': {'traces': urll_traces, 'labels': urll_labels}
    }, cols_names, excluded_indexes, all_cols_names


def extract_feats_stats(cols_names, trials_in):
    columns_maxmin = {}
    for c in range(trials_in.shape[2]):
        col_max = trials_in[:, :, c].max()
        col_min = trials_in[:, :, c].min()
        col_mean = trials_in[:, :, c].mean()
        col_std = trials_in[:, :, c].std()
        columns_maxmin[c] = {'max': col_max, 'min': col_min, 'mean': col_mean, 'std': col_std, 'name': cols_names[c]}
    return columns_maxmin

def normalize_KPIs(columns_maxmin, trials_in, doPrint=False):
    trials_in_norm = trials_in.copy()
    for c, max_min_info in columns_maxmin.items():
        if isinstance(c, int):
            if max_min_info['name'] != 'Timestamp':
                col_max = max_min_info['max']
                col_min = max_min_info['min']
                if not (col_max == col_min):
                    if doPrint:
                        print('Normalizing Col.', max_min_info['name'], ' -- Max', col_max, ', Min', col_min)
                    trials_in_norm[:, :, c] = (trials_in_norm[:, :, c] - col_min) / (col_max - col_min)
                else:
                    trials_in_norm[:, :, c] = 0  # set all data as zero (we don't need this info cause it never changes)
            else:
                if doPrint:
                    print('Skipping normalization of Col. ', max_min_info['name'])

    return trials_in_norm


def normalize_RAW_KPIs(columns_maxmin, trials_in, map_feat2KPI, indexes_to_keep, doPrint=False):
    trials_in_norm = trials_in[:,:,indexes_to_keep].copy()
    for f, max_min_info in columns_maxmin.items():
        if isinstance(f, int):
            c = map_feat2KPI[f]
            col_max = max_min_info['max']
            col_min = max_min_info['min']
            if not (col_max == col_min):
                if doPrint:
                    print('Normalizing Col.', max_min_info['name'], '[', c, '] -- Max', col_max, ', Min', col_min)
                trials_in_norm[:, :, f] = (trials_in_norm[:, :, f] - col_min) / (col_max - col_min)
            else:
                trials_in_norm[:, :, f] = 0  # set all data as zero (we don't need this info cause it never changes)
    return trials_in_norm

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
        ctrl_traces = []


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

                if label == '':
                    print("[load_csv_dataset__multi] Unrecognized label or bad formatting.\n\t"+ trace_filename +"\nSkipping trace..")
                    continue

                ue_csv = os.path.join(m, ueID+"_metrics.csv")
                ue_kpis = pd.read_csv(ue_csv, sep=",")
                ue_kpis = ue_kpis.dropna(how='all', axis='columns')  # remove empty columns
                ds_tree[multi_key][str(ueID)] = {"trace": trace_filename, "label": label, "kpis": ue_kpis }

    return ds_tree


def relative_timestamp(x):
    first_ts = x[0,0] # get first value of first column (Timestamp)
    x[:, 0] -=  first_ts
    return x

def add_first_dim(x):
    return x[None]

class ORANTracesDataset(Dataset):
    def __init__(self, dataset_pkls, key, normalize=True, path='../logs/', norm_par_path="", sanitize=True, relabel_CTRL=False, transform=None, target_transform=None):
        self.obs_input, self.obs_labels = None, None
        self.path = path
        self.norm_params = None

        if norm_par_path != "":
            self.norm_params = pickle.load(open(norm_par_path, 'rb'))

        for p in dataset_pkls:
            dataset = pickle.load(open(os.path.join(self.path, p), 'rb'))
            # let's retrieve the relative normalization parameters for this specific dataset
            # ds_filename = os.path.basename(p)
            # ds_dir = os.path.dirname(p)
            # trials_str, filemarker = ds_filename.split('__')[2:]
            # normp_filename = "cols_maxmin__" + trials_str + "__" + filemarker
            # norm_params = pickle.load(open(os.path.join(self.path, ds_dir, normp_filename), 'rb'))

            if normalize:
                norm_key = 'norm'
            else:
                norm_key = 'no_norm'
            if (self.obs_input is None) and (self.obs_labels is None):
                self.obs_input, self.obs_labels = dataset[key]['samples'][norm_key], dataset[key]['labels']
            else:
                self.obs_input = torch.cat((self.obs_input, dataset[key]['samples'][norm_key]), dim=0)
                self.obs_labels = torch.cat((self.obs_labels, dataset[key]['labels']), dim=0)

        # if needed, relabel the CTRL samples based on computation on the
        # normalization parameters (or just compute it from the dataset)
        self.ctrl_label = 3
        #self.relabel_norm_threshold = min([self.norm_params['info']['norm_dist'][cl]['thr'] for cl in [0, 1, 2]])
        self.relabel_norm_threshold = self.norm_params['info']['norm_dist'][self.ctrl_label]['mean']
        #print("Mean CTRL Norm distance threshold: ", self.relabel_norm_threshold)
        self.relabeled = False
        if relabel_CTRL:
            self.relabel_ctrl_samples()

        # sanitize columns: remove columns with no variation of data (after relabeling)
        if sanitize:
            obs_std = np.std(self.obs_input.numpy(), axis=(0, 1))
            features_to_remove = np.where(obs_std == 0)[0]
            assert np.all(features_to_remove == self.norm_params['info']['exclude_cols_ix']), "Double check that the column to exclude correpsonds to the ones in the common paramters"
            all_feats = torch.arange(0, self.obs_input.shape[-1])
            indexes_to_keep = [i for i in range(len(all_feats)) if i not in self.norm_params['info']['exclude_cols_ix']]
            self.obs_input = self.obs_input[:, :, indexes_to_keep]


        self.sanitized = sanitize

        self.transform = transform
        self.target_transform = target_transform

    def relabel_ctrl_samples(self):
        #ixs_ctrl = self.obs_labels == self.ctrl_label

        assert not(self.norm_params is None) , "Relabeling needs normalization parameters to be passed."
        assert 'info' in self.norm_params.keys(), "Normalization parameters need an 'info' field"
        assert 'mean_ctrl_sample' in self.norm_params['info'].keys(), "Normalization parameters need to have the mean_ctrl_sample pre-computed"

        # exclude column 0 (Timestamp) and 2 (IMSI)
        # NOTE: at this point, we still have all KPIs (unfiltered) in self.obs_input
        # TODO: the "globalnorm" files have filtered parameters now, so this filtering is not needed for mean ctrl template
        include_ixs = set(range(self.obs_input.shape[-1]))
        remove_cols = ['Timestamp', 'IMSI']
        for k, v in self.norm_params.items():
            if isinstance(k, int) and v['name'] in remove_cols:
                include_ixs.remove(k)

        mean_ctrl_sample = self.norm_params['info']['mean_ctrl_sample']
        std_ctrl_sample = self.norm_params['info']['std_ctrl_sample']
        assert mean_ctrl_sample.shape[-1] == len(include_ixs), "Check that features size is the same"

        obs_excludecols = self.obs_input[:, :, list(include_ixs)]
        """
        # NEW METHOD 
        # TODO: needs to be revised
        std_threshold = 2e-2
        zero_std_cols = [col for col in range(std_ctrl_sample.shape[1]) if
                         np.all(std_ctrl_sample[:, col].numpy() < std_threshold)]
        np_obs = obs_excludecols.numpy()
        possible_ctrl_ixs_new = []
        np_mean_ctrl_sample = mean_ctrl_sample.numpy()
        np_std_ctrl_sample = std_ctrl_sample.numpy()
        for sample in tqdm(range(np_obs.shape[0]),desc="Finding CTRL template matches:"):
            col_checks = np.zeros((len(zero_std_cols)))
            for i, c in enumerate(zero_std_cols):
                col_range = np_mean_ctrl_sample[:, c] + (3 * np_std_ctrl_sample[:,c])
                #col_range = np_mean_ctrl_sample[:, c] #- 0.02617632*(1 * np_std_ctrl_sample[:,c])
                #print(col_range)
                if np.all(np_obs[sample,:,c] <= col_range):
                    col_checks[i] = 1
            if np.all(col_checks):
                possible_ctrl_ixs_new.append(sample)

        possible_ctrl_labels = self.obs_labels[possible_ctrl_ixs_new]
        pre_unique_lbls, pre_unique_cnts = np.unique(self.obs_labels, return_counts=True)
        print("Initial # samps. per label (before relabeling)\n\t Labels:", pre_unique_lbls, "Count:", pre_unique_cnts)
        unique_labels, unique_counts = np.unique(possible_ctrl_labels, return_counts=True)
        print("\tLabels that match CTRL mean template", unique_labels,
              "\n\tNum of matching samples per label:", unique_counts)
        # plt.bar(unique_labels, unique_counts, tick_label=unique_labels)
        # plt.show()
        count_relabel = 0
        for cix in possible_ctrl_ixs_new:
            if self.obs_labels[cix] != self.ctrl_label:
                # print(cix, self.obs_labels[cix], '-> 3')
                self.obs_labels[cix] = self.ctrl_label
                count_relabel += 1
        print("Tot. samples relabeled (for every class):", count_relabel)
        """

        # OLD METHOD
        # compute euclidean distance between samples of other classes and mean ctrl sample
        norm = np.linalg.norm(obs_excludecols - mean_ctrl_sample, axis=(1, 2))
        # here we get the indexes for all samples that have a norm (i.e. Euclidean distance) less than a given
        # threshold, which should correspond to CTRL (i.e. silent) traffic portions. Note that the lower the threshold,
        # the more conservative is the relabeling. This threshold is computed based on the distribution of euclidean
        # distances computed between the mean CTRL sample (assuming they look very similar)
        # and every sample of every other class
        possible_ctrl_ixs = norm < self.relabel_norm_threshold
        possible_ctrl_labels = self.obs_labels[possible_ctrl_ixs]
        pre_unique_lbls, pre_unique_cnts = np.unique(self.obs_labels, return_counts=True)
        print("Initial # samps. per label (before relabeling)\n\t Labels:", pre_unique_lbls, "Count:", pre_unique_cnts)
        unique_labels, unique_counts = np.unique(possible_ctrl_labels, return_counts=True)
        print("\tLabels that contain norm < threshold", unique_labels, "\n\tNum of samples per label with norm < threshold:", unique_counts)
        #plt.bar(unique_labels, unique_counts, tick_label=unique_labels)
        #plt.show()
        
        count_relabel = 0
        for ix, isPossibleCTRL in enumerate(possible_ctrl_ixs):
            if isPossibleCTRL and self.obs_labels[ix] != self.ctrl_label:
                #print(ix, self.obs_labels[ix], '-> 3')
                self.obs_labels[ix] = self.ctrl_label
                count_relabel += 1
        print("Tot. samples relabeled (for every class):", count_relabel)

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

def safe_pickle_dump(filepath, myobj):
    yes_choice = ['yes', 'y']
    to_save = True
    if os.path.isfile(filepath):
        user_input = input("File " + filepath + " exists already. Overwrite? [y/n]")
        if not (user_input.lower() in yes_choice):
            to_save = False
    if to_save:
        pickle.dump(myobj, open(filepath, 'wb'))

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
    parser.add_argument("--already_gen", action='store_true', default=False, help="[DEBUG] Use this flag for pre-generated dataset(s) that are only needed to compute new statistics.")
    parser.add_argument("--exp_name", default='', help="Name of this experiment")
    parser.add_argument("--cp_path", help='Path to save/load checkpoint and training/dataset logs')
    parser.add_argument("--ds_pkl_paths", default="", help="(--already-gen) specify origin pkl file.")
    parser.add_argument("--normp_pkl", default="", help="(--already-gen) specify origin pkl file.")
    parser.add_argument("--exclude_cols", nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8, 14, 22, 27, 28, 29], help="(--already-gen) specify origin pkl file.")

    args, _ = parser.parse_known_args()

    from visual_xapp_inference import classmap

    path = args.ds_path
    trials = args.trials
    trials_multi = args.trials_multi
    ctrl_class = 3
    isPlot = True
    norm_threshold = None
    logdir = os.path.join(args.cp_path, args.exp_name)
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)


    datasets = []

    if not args.already_gen:
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

            dataset_path = os.path.join(pickle_ds_path, 'dataset__' + args.mode + '__' +file_suffix + zeros_suffix + '.pkl')
            safe_pickle_dump(dataset_path, dataset)
            # save separately maxmin normalization parameters for each column/feature
            norm_param_path = os.path.join(pickle_ds_path,'cols_maxmin__'+file_suffix+'.pkl')
            safe_pickle_dump(norm_param_path, cols_maxmin)

            datasets.append( (dataset, cols_maxmin, dataset_path, norm_param_path) )

    else:
        # DEBUG: in case you have already generated the datasets and
        # want just load it instead of recompute normalization parameters
        if args.ds_pkl_paths != "":
            # TODO add possibiity to specify a list of pkl files instead of default paths (to allow multi UE also etc.)
            assert (args.normp_pkl != ""), "[already_gen] Norm. params .pkl file has to be passed along dataset .pkl(s)"
            dataset_path = args.ds_pkl_paths
            pickle_ds_path = args.normp_pkl
            dataset = pickle.load(open(dataset_path, "rb"))
            cols_maxmin = pickle.load(open(pickle_ds_path, "rb"))
            datasets.append((dataset, cols_maxmin, dataset_path, pickle_ds_path))

        else:
            for data_type in args.data_type:
                trials = trials_multi if "multi" in data_type else trials
                file_suffix = ''
                for t in trials:
                    file_suffix += t
                    if t != trials[-1]:
                        file_suffix += '_'

                file_suffix += '__slice' + str(args.slicelen) + '_' + data_type
                file_suffix += '_' + args.filemarker if args.filemarker else ''
                zeros_suffix = '_ctrlcorrected_' if args.check_zeros else ''
                pickle_ds_path = os.path.join(path, "Multi-UE") if "multi" in data_type else os.path.join(path, "SingleUE")

                dataset_path = os.path.join(pickle_ds_path,
                                            'dataset__' + args.mode + '__' + file_suffix + zeros_suffix + '.pkl')

                dataset = pickle.load(open(dataset_path, "rb"))
                # load maxmin normalization parameters for each column/feature
                norm_param_path = os.path.join(pickle_ds_path, 'cols_maxmin__' + file_suffix + '.pkl')
                cols_maxmin = pickle.load(open(norm_param_path, "rb"))

                datasets.append((dataset, cols_maxmin, dataset_path, norm_param_path))

    # if multiple data types (i.e. singleUE, multiUE) have been processed,
    # let's define a common set of normalization/sanitization parameters

    # first, let's find a common set of columns to keep (i.e. the ones that have varying values)
    # and their common min max parameters for normalization
    common_normp = {}
    novar_keys_raw = []
    novar_keys = []
    common_keys = set()
    [common_keys.update(
            set([k for k in kpi_p.keys() if isinstance(k, int)])
        )
        for _, kpi_p, _, _ in datasets]

    # verify that the raw column names are all the same  for all datasets
    raw_col_names = datasets[0][1]['info']['raw_cols_names']
    assert all([np.all(kpi_p['info']['raw_cols_names'] == raw_col_names) for _, kpi_p, _, _ in
                datasets]), 'Ensure that raw kpis names are the same for all datasets'
    exclude_cols_ix = datasets[0][1]['info']['exclude_cols_ix']
    assert all([np.all(kpi_p['info']['exclude_cols_ix'] == exclude_cols_ix) for _, kpi_p, _, _ in
                datasets]), 'Ensure that excluded kpi indexes are the same for all datasets'

    for k in common_keys:   # for each common key/kpi featurepickle_ds_path
        # verify that the key names are the same for all datasets
        first_ds_kpiname = datasets[0][1][k]['name']
        assert all([kpi_p[k]['name'] == first_ds_kpiname for _, kpi_p, _, _ in datasets]), 'Ensure that kpis and names have the same order for all datasets'

        # obtain the overall max/min values
        common_normp[k] = {
            'name': first_ds_kpiname,
            'max': max([kpi_p[k]['max'] for _, kpi_p, _, _ in datasets]),     # get max of maxes in that specific key/feature
            'min': min([kpi_p[k]['min'] for _, kpi_p, _, _ in datasets])      # get min of mins in that specific key/feature
        }
        # annotate the features indexes that needs to be dropped when re-normalizing and sanitizing the data
        if common_normp[k]['max'] == common_normp[k]['min']:
            for ix, name in enumerate(raw_col_names):    # use the names of first kpi params in the dataset (since they should be all the same)
                if common_normp[k]['name'] == name:
                    c = ix
                    break
            novar_keys_raw.append(c)
            novar_keys.append(k)

    # let's also exclude the indexes from normalization parameters to reflect filtered features
    common_normp_filtd = {}
    feat_count = 0
    for k in common_keys:
        if not(k in novar_keys):
            common_normp_filtd[feat_count] = common_normp[k].copy()
            feat_count += 1
    common_normp_filtd['info'] = {'exclude_cols_ix': list(set(novar_keys_raw).union(set(exclude_cols_ix)))}

    common_normp = common_normp_filtd   # substitute the common norm parameters with newly filtered ones

    filemarker = '__' + args.filemarker if args.filemarker else ''
    global_norm_path = os.path.join(path, 'global__cols_maxmin'+filemarker+'_slice'+str(args.slicelen)+'.pkl')

    # until now, we have retained the columns that are not excluded, but we still haven't removed
    # the ones without variation (i.e. std = 0).
    # we should remove those columns here as well before continuing computation of ctrl template
    filtKPI_keep = list(set(range(len(common_keys))).difference(novar_keys))

    for ds, kpi_p, ds_path, kpi_p_path in datasets:
        for t in ['train', 'valid']:
            ds[t]['samples']['norm'] = ds[t]['samples']['norm'][:, :, filtKPI_keep]
            ds[t]['samples']['no_norm'] = ds[t]['samples']['no_norm'][:, :, filtKPI_keep]
        safe_pickle_dump(os.path.splitext(ds_path)[0] + "__globalnorm.pkl", ds)

    # then let's create an alternative version of the datasets normalized with global paramters and save it in a separate file
    for ds, kpi_p, ds_path, kpi_p_path in datasets:
        print('---- Normalizing Training set ----')
        trials_train_globalminmax = normalize_KPIs(common_normp, ds['train']['samples']['no_norm'].numpy(), doPrint=True) # filter out the additional columns with no variance
        ds['train']['samples']['norm'] = torch.Tensor(trials_train_globalminmax)
        print('---- Normalizing Validation set ----')
        trials_valid_globalminmax = normalize_KPIs(common_normp, ds['valid']['samples']['no_norm'].numpy(), doPrint=True) # filter out the additional columns with no variance
        ds['valid']['samples']['norm'] = torch.Tensor(trials_valid_globalminmax)
        ds['info'] = {'global_norm_path': global_norm_path}



    # second, let's compute an average CTRL traffic template to be used across multiple datasets
    # IMPORTANT: we have to use also the validation set for this and after the new normalization
    all_ctrl = None
    for ds, kpi_p, ds_path, kpi_p_path in datasets:
        for t in ['train', 'valid']:
            ixs_ctrl = ds[t]['labels'] == ctrl_class
            if torch.any(ixs_ctrl):
                if all_ctrl is None:
                    all_ctrl = ds[t]['samples']['norm'][ixs_ctrl]
                else:
                    all_ctrl = torch.cat((all_ctrl, ds[t]['samples']['norm'][ixs_ctrl]), dim=0)


    if (not (all_ctrl is None)) and (all_ctrl.shape[0] > 0):
        mean_ctrl_sample = torch.mean(all_ctrl, dim=0)
        std_ctrl_sample = torch.std(all_ctrl, dim=0)

        common_normp['info']['mean_ctrl_sample'] = mean_ctrl_sample
        common_normp['info']['std_ctrl_sample'] = std_ctrl_sample

        common_normp['info']['norm_dist'] = {}
        x_axis = np.arange(0, 15, 0.01)

        for cl in [0, 1, 2, 3]:
            print("Difference of mean class ctrl (4) with class", cl)

            all_sample = None
            for ds, kpi_p, ds_path, kpi_p_path in datasets:
                for t in ['train', 'valid']:
                    ixs_ctrl = ds[t]['labels'] == cl
                    if torch.any(ixs_ctrl):
                        if all_sample is None:
                            all_sample = ds[t]['samples']['norm'][ixs_ctrl]
                        else:
                            all_sample = torch.cat((all_sample, ds[t]['samples']['norm'][ixs_ctrl]), dim=0)

            norm = np.linalg.norm(all_sample - mean_ctrl_sample, axis=(1, 2))
            print("Mean norm", np.mean(norm), "Std.", np.std(norm))
            common_normp['info']['norm_dist'][cl] = {'mean': np.mean(norm), 'std': np.std(norm)}

        relabel_method = "conservative" # or progressive :)
        if isPlot:
            import matplotlib.pyplot as plt
        # compute euclidean distance between samples of other classes and mean ctrl sample

        # set the threshold for probability of observing a ctrl message based on its norm distance from template
        ctrl_distrib = normal.pdf(x_axis, common_normp['info']['norm_dist'][3]['mean'], common_normp['info']['norm_dist'][3]['std'])
        if isPlot:
            plt.figure(figsize=(7.5, 4.2))
            plt.rc('font', size=14)  # Main text
            plt.rc('axes', titlesize=14)  # Title
            plt.rc('axes', labelsize=14)  # X and Y axis labels
            plt.rc('xtick', labelsize=12)  # X-axis tick labels
            plt.rc('ytick', labelsize=12)  # Y-axis tick labels
            plt.rc('legend', fontsize=14)  # Legend font size
            plt.rc('figure', titlesize=10)  # Figure title

            plt.plot(x_axis, ctrl_distrib, label='ctrl')
        for cl, traff_name in {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}.items():
            this_mean, this_std = common_normp['info']['norm_dist'][cl]['mean'], common_normp['info']['norm_dist'][cl]['std']
            this_distrib = normal.pdf(x_axis, this_mean, this_std)
            if relabel_method == "progressive":
                # here we compare the distribution curves and pick the threshold as the point in the distribution
                # where the probability of being ctrl is less than all the others
                thr_ix = np.where(ctrl_distrib < this_distrib)[0][0]
                thr_val = x_axis[thr_ix]
            else:
                # here we adopt a more conservative method. For each distribution of classes (except CTRL),
                # we subtract 1 standard deviation from its mean and use that as threshold
                thr_val = this_mean - (1 * this_std)

            common_normp['info']['norm_dist'][cl]['thr'] = thr_val # save threshold value
            print("Threshold class ", cl, ":", thr_val)
            if isPlot:
                if traff_name == "eMBB":
                    linestyle='--'
                elif traff_name == 'mMTC':
                    linestyle='-.'
                elif traff_name == 'URLLC':
                    linestyle=':'
                plt.plot(x_axis, this_distrib, linestyle, label=traff_name )

        # find min threshold
        #min_threshold = min([common_normp['info']['norm_dist'][cl]['thr'] for cl in [0, 1, 2]])
        min_threshold = common_normp['info']['norm_dist'][3]['mean']
        norm_threshold = min_threshold

        if isPlot:

            plt.legend()
            plt.xlabel('Euclidean distance from mean ctrl template')
            plt.ylabel('Probability density')
            plt.axvline(x=min_threshold, ls='-', c='gray')
            plt.savefig(os.path.join(logdir, 'normDiff_distr.png'))
            plt.clf()

        if isPlot:
            plt.imshow(mean_ctrl_sample)
            plt.colorbar()
            plt.savefig(os.path.join(logdir, 'mean_ctrl_sample.png'))
            plt.clf()
            #plt.imshow(std_ctrl_sample)
            #plt.colorbar()
            #plt.show()

        all_sample_noctrl = None
        all_labels_noctrl = None
        for ds, kpi_p, ds_path, kpi_p_path in datasets:
            for t in ['train', 'valid']:
                ixs_ctrl = ds[t]['labels'] != ctrl_class
                if torch.any(ixs_ctrl):
                    if all_sample_noctrl is None and all_labels_noctrl is None:
                        all_sample_noctrl = ds[t]['samples']['norm'][ixs_ctrl]
                        all_labels_noctrl = ds[t]['labels'][ixs_ctrl]
                    else:
                        all_sample_noctrl = torch.cat((all_sample_noctrl, ds[t]['samples']['norm'][ixs_ctrl]), dim=0)
                        all_labels_noctrl = torch.cat((all_labels_noctrl, ds[t]['labels'][ixs_ctrl]), dim=0)
        # show a barplot representing the amount of samples that should be relabeled
        obs_excludecols = all_sample_noctrl
        norm = np.linalg.norm(obs_excludecols - mean_ctrl_sample, axis=(1, 2))
        possible_ctrl_ixs = norm < norm_threshold # dynamic threshold based on slice length and datasets combinations
        possible_ctrl_labels = all_labels_noctrl[possible_ctrl_ixs].numpy()
        unique_labels, unique_counts = np.unique(possible_ctrl_labels, return_counts=True)
        print("Unique labels", unique_labels, " count", unique_counts)
        if isPlot:
            plt.bar(unique_labels, unique_counts, tick_label=unique_labels)
            plt.savefig(os.path.join(logdir, 'num_relabel_noctrl.png'))
            plt.clf()

        sample_plots = all_sample_noctrl[possible_ctrl_ixs].numpy()
        label_plots = all_labels_noctrl[possible_ctrl_ixs].numpy()
        norm_plots = norm[possible_ctrl_ixs]
        """
        os.makedirs(os.path.join(path, 'img'), exist_ok=True)
        for s in range(sample_plots.shape[0]):
            plt.imsave(os.path.join(path, 'img', "lbl" + str(label_plots[s]) + "__relab" + str(s) + "_th_" + str(
                norm_plots[s]) + "_sample_noctrl.png"), sample_plots[s])
        """
    # lastly, let's save the new global parameters
    safe_pickle_dump(global_norm_path, common_normp)

