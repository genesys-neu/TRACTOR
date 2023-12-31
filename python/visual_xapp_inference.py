import numpy
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


classmap = {'embb': 0, 'mmtc': 1, 'urll': 2, 'ctrl': 3}
colormap = {0: '#D97652', 1: '#56A662', 2: '#BF4E58', 3: '#8172B3'}
hatchmap = {0: '/', 1: '\\', 2: '//', 3: '.' }


import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

import torch


def plot_trace_class(output_list_kpi, output_list_y, img_path, slice_len, head=0, save_plain_img=False, postfix='', folder_postfix='', colormap = {0: '#D97652', 1: '#56A662', 2: '#BF4E58', 3: '#8172B3'}, hatchmap = {0: '/', 1: '\\', 2: '//', 3: '.' }):
    imgout = np.array(output_list_kpi).T
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(50, 5))
    # Display the image
    pos = ax.imshow(imgout, extent=[0, len(output_list_kpi), 0, imgout.shape[0]], aspect='auto', vmin=0., vmax=1.)
    imgs_path = os.path.join(img_path + '/imgs_'+folder_postfix, 'slice' + str(slice_len))
    os.makedirs(imgs_path, exist_ok=True)
    if save_plain_img:
        #fig.colorbar(pos)
        plt.savefig(os.path.join(imgs_path, 'outputs_' + os.path.basename(img_path) + 's' + str(
            head - len(output_list_kpi)) + '_e' + str(
            head) + '__plain.png'))

    # add white background
    ax.imshow(np.ones(imgout.shape), extent=[0, len(output_list_kpi), 0, imgout.shape[0]], cmap='bone', aspect='auto', vmin=0., vmax=1.)
    plt.rcParams['hatch.linewidth'] = 2.0  # previous svg hatch linewidth
    lbl_old = None
    rect_len = 0
    for ix, label in enumerate(output_list_y):
        if isinstance(label, int) or isinstance(label, numpy.int64):
            lbl = label
        elif isinstance(label, torch.Tensor):
            lbl = label.numpy()[0]
        # Create a Rectangle patch
        #print(lbl)
        if lbl_old is None:
            lbl_old = lbl
            rect_len = 1
            slicestart_ix = ix
            continue
        else:

            if not(ix == (len(output_list_y)-1)) and lbl_old == lbl:  # if the same label has been assigned as before
                rect_len += 1   # increase size of patch by 1
                lbl_old = lbl   # update the prev class label
                continue    # skip to next input sample without printing
            else:
                rect_len += slice_len - 1   # we set the remainder rectangle length based on the slice length
                # proceed to printing the Rectangle up until the previous sample

        #Here we plot the rectangle for up until the previous block
        if hatchmap is None:
            rect = patches.Rectangle((slicestart_ix, 0), rect_len, imgout.shape[0], linewidth=1, edgecolor=colormap[lbl_old], facecolor=colormap[lbl_old], alpha=1)
        else:
            rect = patches.Rectangle((slicestart_ix, 0), rect_len, imgout.shape[0], hatch=hatchmap[lbl_old], edgecolor='white', facecolor=colormap[lbl_old], linewidth=0)
        # Add the patch to the Axes
        ax.add_patch(rect)
        # then we reset the info for the next block
        lbl_old = lbl
        rect_len = 1    # reset rectangle len
        slicestart_ix = ix  # set the start for the next rectangle

    plt.savefig(os.path.join(imgs_path, 'outputs_' + os.path.basename(img_path) + 's' + str(head - len(output_list_kpi)) + '_e' + str(
        head) + postfix + '.png'))
    plt.clf()


def process_norm_params(all_feats_raw, colsparam_dict):
    all_feats = np.arange(0, all_feats_raw)
    exclude_param = colsparam_dict['info']['exclude_cols_ix']
    colsparams = {key: value for key, value in colsparam_dict.items() if isinstance(key, int)}
    indexes_to_keep = np.array([i for i in range(len(all_feats)) if i not in exclude_param])
    # we obtain num of input features this from the normalization/relabeling info
    num_feats = len(indexes_to_keep)
    slice_len = colsparam_dict['info']['mean_ctrl_sample'].shape[0]
    # create a map from indexes after features/KPIs filtering and original feature index
    # to retrieve normalization parameters
    map_feat2KPI = dict(zip(np.arange(0, len(indexes_to_keep)), indexes_to_keep))
    print("INFO:\n",
          "\tSlice len. (T) =", slice_len, "\tNum. Feats (K)=", num_feats, "\n",
          "\tIndexes to be kept:", repr(indexes_to_keep), "\n",
          "\tIndexes to be excluded:", repr(exclude_param), "\n",
          "\tFeature-to-KPI map", repr(map_feat2KPI), "\n",
          "Column params for normalization:\n", repr(colsparams)
          )
    return colsparams, indexes_to_keep, map_feat2KPI, num_feats, slice_len, colsparam_dict['info']['mean_ctrl_sample']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", required=True, help="Path containing the classifier output files for re-played traffic traces")
    parser.add_argument("--mode", choices=['pre-comp', 'inference'], default='pre-comp', help="Specify the type of file format we are trying to read.")
    parser.add_argument("--slicelen", choices=[4, 8, 16, 32, 64], type=int, default=32, help="Specify the slicelen to determine the classifier to load")
    parser.add_argument("--model_path", help="Path to TRACTOR model to load."  )
    parser.add_argument("--norm_param_path", default="", help="Normalization parameters path.")
    parser.add_argument("--model_type", default="Tv1", choices=['CNN', 'Tv1', 'Tv1_old', 'Tv2', 'ViT'], help="Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)")
    parser.add_argument("--Nclasses", default=4, help="Used to initialize the model")
    parser.add_argument("--dir_postfix", default="", help="This is appended to the name of the output folder for images and text")
    parser.add_argument("--CTRLcheck", action='store_true', default=False, help="At test time (inference), it will compare the sample with CTRL template to determine if its a correct CTRL sample")
    parser.add_argument("--chZeros", action='store_true', default=False, help="[Deprecated] At test time, don't count the occurrences of ctrl class")

    args, _ = parser.parse_known_args()

    from ORAN_models import ConvNN, TransformerNN, TransformerNN_v2, TransformerNN_old
    from ORAN_dataset import normalize_RAW_KPIs

    PATH = args.trace_path
    if PATH[-1] == '/':  # due to use of os.basename
        PATH = PATH[:-1]

    check_zeros = args.chZeros
    check_ctrl_tpl = args.CTRLcheck

    if args.mode == 'pre-comp':
        slice_len = args.slicelen
        output_list_kpi = []
        output_list_kpi_raw = []
        output_list_y = []
        head = 0

        pkl_list = glob.glob(os.path.join(PATH, 'class_output_*.pkl'))
        pkl_list.sort(key=natural_keys)

        for ix, p in enumerate(pkl_list):
            TRACTOR_inout = pickle.load(open(p, 'rb'))

            kpis = TRACTOR_inout['input']
            class_out = TRACTOR_inout['label']

            # construct raw input (note, we have to skip the first T - 1 samples
            kpis_raw = TRACTOR_inout['input_raw']

            """
            for k in classmap.keys():
                if k in PATH:
                    correct_class = classmap[k]
        
            co = class_out.numpy()[0]
            print('Class', co)
            plt.pcolor(kpis, vmin=0., vmax=1.)
            plt.colorbar()
            plt.title('Inferred class:'+str(co))
            plt.savefig(os.path.join(PATH, 'fig_input_'+str(ix)+'_c'+str(co)+'.png'))
            plt.clf()
            if co == correct_class:
                num_correct += 1
            """

            if ix == 0:
                old_kpis = kpis.copy()  # dim: [slice_len, num_feats]
                old_kpis_raw = kpis_raw.copy() # dim: [num_feats_raw,]
                for i in range(slice_len):  # add samples for all T
                    output_list_kpi.append(kpis[i,:])

                output_list_kpi_raw.append(kpis_raw[np.newaxis, :]) # add single RAW line (TODO atm we only have saved it like this)
                head = len(output_list_kpi)
            elif ix > 0:
                if np.all(kpis[0:slice_len-1, :] == old_kpis[1:slice_len, :]):
                    print('Kpis', ix, ' contiguous')
                    output_list_kpi.append(kpis[-1, :]) # add only the last element to visualization output
                    output_list_kpi_raw.append(kpis_raw[np.newaxis, :]) # add the current KPI (TODO note that there is a +T offset atm)
                    head += 1
                else:
                    print('Kpis', ix, 'NOT contiguous')
                    # first, let's plot everything until now and empty the output buffer
                    plot_trace_class(output_list_kpi, output_list_y, PATH, slice_len, head, save_plain_img=True)
                    pickle.dump({'input_trace': np.array(output_list_kpi), 'raw_trace': np.array(output_list_kpi_raw)}, open(PATH+'/replay_kpis__'+ os.path.basename(PATH) + 's' +
                                str(head - len(output_list_kpi)) + '_e' + str(head) +'.pkl', 'wb'))
                    # reset output lists
                    output_list_kpi = []
                    output_list_kpi_raw = []
                    output_list_y = []

                    for i in range(slice_len):
                        output_list_kpi.append(kpis[i, :])

                    output_list_kpi_raw.append(kpis_raw[np.newaxis, :])
                    head += len(output_list_kpi)

                old_kpis = kpis.copy()
                old_kpis_raw = kpis_raw.copy()

            output_list_y.append(class_out)

        # if there's data in the buffer
        if len(output_list_kpi) > 0 and len(output_list_kpi_raw) > 0:
            # let's print the accumulated KPI inputs and relative outputs
            imgout = np.array(output_list_kpi).T # TODO: are we sure the Transpose is doing the right thing??
            plot_trace_class(output_list_kpi, output_list_y, PATH, slice_len, head, save_plain_img=True)
            pickle.dump({'input_trace': np.array(output_list_kpi), 'raw_trace': np.array(output_list_kpi_raw)}, open(
                PATH + '/replay_kpis__' + os.path.basename(PATH) + 's' + str(head - len(output_list_kpi)) + '_e' + str(
                    head) + '.pkl', 'wb'))

            # TODO count occurrences of each label

        """
        print('Correct % ', num_correct/len(pkl_list)*100)
        
        from python.ORAN_dataset import *
        dsfile = '/home/mauro/Research/ORAN/traffic_gen2/logs/dataset__emuc__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice8_wCQI.pkl'
        dspath = '/home/mauro/Research/ORAN/traffic_gen2/logs/'
        ds_train = ORANTracesDataset(dsfile, key='train', normalize=True, path=dspath)
        
        max_samples = 50
        train_samples = {c: [] for c in range(4)}
        for samp, lbl in ds_train:
            if all([len(train_samples[c]) == 50 for c in train_samples.keys()]):
                break
            c = int(lbl.numpy())
            if len(train_samples[c]) < max_samples:
                train_samples[c].append((samp, lbl))
        
        for c, samples in train_samples.items():
            for ix, s in enumerate(samples):
                plt.pcolor(s[0], vmin=0., vmax=1.)
                plt.colorbar()
                plt.title('Real class:'+str(c))
                plt.savefig(os.path.join('train_samps/', 'train__fig_input_'+str(ix)+'_c'+str(c)+'.png'))
                plt.clf()
        
        """

    elif args.mode == 'inference':
        pos_enc = False  # not supported at the moment

        if args.model_type is not None:
            if args.model_type == 'Tv1':
                model_type = TransformerNN
            elif args.model_type == 'Tv1_old':
                model_type = TransformerNN_old
            elif args.model_type == 'Tv2':
                model_type = TransformerNN_v2
            elif args.model_type == 'ViT':
                # transformer = ViT
                print("Transformer type " + args.transformer + " is not yet supported.")
                exit(-1)
            elif args.model_type == 'CNN':
                model_type = ConvNN

        torch_model_path = args.model_path
        norm_param_path = args.norm_param_path

        Nclass = args.Nclasses
        all_feats_raw = 31

        colsparam_dict = pickle.load(open(norm_param_path, 'rb'))
        #relabel_norm_threshold = min([colsparam_dict['info']['norm_dist'][cl]['thr'] for cl in [0, 1, 2]])
        relabel_norm_threshold = colsparam_dict['info']['norm_dist'][3]['mean']
        (colsparams,
         indexes_to_keep,
         map_feat2KPI,
         num_feats,
         slice_len,
         mean_ctrl_sample) = process_norm_params(all_feats_raw, colsparam_dict)

        # initialize the KPI matrix
        kpi = []
        last_timestamp = 0
        curr_timestamp = 0

        # initialize the ML model
        print('Init ML model...')
        if model_type in [TransformerNN, TransformerNN_v2]:
            model = model_type(classes=Nclass, slice_len=slice_len, num_feats=num_feats, use_pos=pos_enc, nhead=1,
                               custom_enc=True)
        elif model_type == TransformerNN_old:
            model = model_type(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
        elif model_type == ConvNN:
            model = model_type(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
        else:
            # TODO
            print("ViT/other model is not yet supported. Aborting.")
            exit(-1)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.load_state_dict(torch.load(torch_model_path, map_location='cuda:0')['model_state_dict'])
        else:
            device = 'cpu'
            model.load_state_dict(torch.load(torch_model_path, map_location='cpu')['model_state_dict'])
        model.to(device)
        model.eval()

        pkl_list = glob.glob(os.path.join(PATH, 'replay*.pkl'))
        for ix, p in enumerate(pkl_list):
            replay_trace_dict = pickle.load(open(p, 'rb'))

            kpis = replay_trace_dict['input_trace']
            kpis_raw = replay_trace_dict['raw_trace']

            output_list_y = []
            if 'embb' in os.path.basename(p):
                correct_class = classmap['embb']
            elif 'mmtc' in os.path.basename(p):
                correct_class = classmap['mmtc']
            elif 'urll' in os.path.basename(p):
                correct_class = classmap['urll']
            else:
                correct_class = classmap['ctrl']

            num_correct = 0
            num_samples = 0
            num_verified_ctrl = 0
            num_heuristic_ctrl = 0
            filt_kpi_offset = slice_len # TODO we are only saving the last raw KPI in xapp code
                                        #   momentarily... we skip the first "slice_len" samples
            txt_output = ""
            for t in range(kpis_raw.shape[0]):  # iterate over the size of raw kpis and account for the offset of filtered ones
                #print('kpis_raw[',t,']')
                if t + args.slicelen < kpis_raw.shape[0]:
                    input_sample = kpis[t+filt_kpi_offset:t+filt_kpi_offset + args.slicelen]
                    input_sample_raw = np.squeeze( kpis_raw[t:t + args.slicelen] )
                    kpi_filt = input_sample_raw[:,indexes_to_keep]

                    for f in range(kpi_filt.shape[1]):
                        if np.any(kpi_filt[:, f] > colsparams[f]['max']) or np.any(kpi_filt[:, f] < colsparams[f]['min']):
                            # print("Clipping ", colsparams[c]['min'], "< x <", colsparams[c]['max'])
                            kpi_filt[:, f] = np.clip(kpi_filt[:, f], colsparams[f]['min'], colsparams[f]['max'])

                        # print('Un-normalized vector'+repr(kpi_filt[:, f]))
                        kpi_filt[:, f] = (kpi_filt[:, f] - colsparams[f]['min']) / (
                                    colsparams[f]['max'] - colsparams[f]['min'])
                        # print('Normalized vector: '+repr(kpi_filt[:, f]))

                    input = torch.Tensor(kpi_filt[np.newaxis, :, :])
                    input = input.to(device)  # transfer input data to GPU
                    pred = model(input)
                    class_ix = pred.argmax(1)
                    co = int(class_ix.cpu().numpy()[0])
                    output_list_y.append(co)
                    if check_zeros:
                        zeros = (input_sample == 0).astype(int).sum(axis=1)
                        if (zeros > 10).all():
                            num_heuristic_ctrl += 1
                            if co == classmap['ctrl']:
                                num_verified_ctrl += 1  #  classifier and heuristic for control traffic agrees
                                continue #skip this sample

                    elif check_ctrl_tpl:

                        # exclude column 0 (Timestamp) and 2 (IMSI)
                        # NOTE: at this point, we still have all KPIs (unfiltered) in mean_ctrl_sample
                        #include_ixs = set(range(all_feats_raw-1)) if colsparam_dict[0] != 'Timestamp' else set(range(all_feats_raw))
                        """
                        remove_cols = ['Timestamp', 'IMSI']
                        for k, v in colsparam_dict.items():
                            if isinstance(k, int) and v['name'] in remove_cols:
                                include_ixs.remove(k)
                        """
                        assert mean_ctrl_sample.shape[-1] == len(indexes_to_keep), "Check that features size is the same"
                        #if colsparam_dict[0] != 'Timestamp':
                        #    input_sample_raw = input_sample_raw[:, 1:]  # remove the first column (Timestamp)

                        input_sample_filtd_norm = normalize_RAW_KPIs(columns_maxmin=colsparam_dict,
                                                                   trials_in=input_sample_raw[np.newaxis,:],
                                                                   map_feat2KPI=map_feat2KPI,
                                                                   indexes_to_keep=indexes_to_keep,
                                                                   doPrint=False)
                        # compute Euclidean distance between samples of other classes and mean ctrl sample
                        obs_excludecols = np.squeeze(input_sample_filtd_norm)

                        norm = np.linalg.norm(obs_excludecols - np.array(mean_ctrl_sample))
                        # here we measure the norm (i.e. Euclidean distance) with mean_ctrl_sample is less than a given
                        # threshold, which should correspond to CTRL (i.e. silent) traffic portions. Note that the lower the threshold,
                        # the more conservative is the relabeling. This threshold is computed based on the distribution of euclidean
                        # distances computed between the mean CTRL sample (assuming they look very similar)
                        # and every sample of every other class
                        if norm < relabel_norm_threshold:
                            num_heuristic_ctrl += 1
                            # plt.imshow(obs_excludecols)
                            # plt.title("Pred: "+str(co))
                            # plt.colorbar()
                            # plt.show()
                            if co == classmap['ctrl']:
                                num_verified_ctrl += 1  # classifier and heuristic for control traffic agrees
                                continue  # skip this sample

                    num_correct += 1 if (co == correct_class) else 0
                    num_samples += 1

            if num_samples > 0:
                mypost = '_cnn_' if isinstance(model, ConvNN) else '_trans_'
                mypost += '_slice' + str(args.slicelen)
                mypost += '_chZero' if check_zeros else ''
                mypost += '_whitebg'
                plot_trace_class(kpis, output_list_y, PATH, args.slicelen, folder_postfix=args.dir_postfix, postfix=mypost, save_plain_img=True, hatchmap=None)
                mypost += '_hatch'
                plot_trace_class(kpis, output_list_y, PATH, args.slicelen, folder_postfix=args.dir_postfix, postfix=mypost, save_plain_img=True)

                print("Correct classification for traffic type (%): ", (num_correct / num_samples)*100., "num correct =", num_correct, ", num classifications =", num_samples)
                txt_output += "Correct classification for traffic type (%): "+str((num_correct / num_samples)*100.)+" num correct = "+str(num_correct)+", num classifications ="+ str(num_samples)+"\n"

            if check_zeros or check_ctrl_tpl:
                unique, counts = np.unique(output_list_y, return_counts=True)
                count_class = dict(zip(unique, counts))
                print(count_class)
                txt_output += str(count_class)+"\n"
                if 3 in count_class.keys():
                    if num_heuristic_ctrl > 0:
                        print("Percent of verified ctrl (through heuristic): ", (num_verified_ctrl / num_heuristic_ctrl)*100., "num verified =", num_verified_ctrl, ", num heuristic matches =", num_heuristic_ctrl )
                        txt_output += "Percent of verified ctrl (through heuristic): " +str((num_verified_ctrl / num_heuristic_ctrl)*100.)+ "num verified = "+str(num_verified_ctrl)+", num heuristic matches = "+str(num_heuristic_ctrl)+"\n"
                    else:
                        print("No ctrl captured by the heuristic")
                        txt_output += "No ctrl captured by the heuristic"+"\n"
                else:
                    print("No ctrl captured by the heuristc")
                    txt_output += "No ctrl captured by the heuristc"+"\n"

            if txt_output != "":
                imgs_path = os.path.join(PATH, 'imgs_'+args.dir_postfix, 'slice' + str(slice_len))
                with open(os.path.join(imgs_path, "txt_output.log"), "w") as txt_file:
                    txt_file.write(txt_output)




