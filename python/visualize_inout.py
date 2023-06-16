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
from ORAN_models import ConvNN, TransformerNN, TransformerNN_v2

def plot_trace_class(output_list_kpi, output_list_y, img_path, slice_len, head=0, save_plain_img=False, postfix='', colormap = {0: '#D97652', 1: '#56A662', 2: '#BF4E58', 3: '#8172B3'}, hatchmap = {0: '/', 1: '\\', 2: '//', 3: '.' }):
    imgout = np.array(output_list_kpi).T
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(50, 5))
    # Display the image
    pos = ax.imshow(imgout, extent=[0, len(output_list_kpi), 0, imgout.shape[0]], aspect='auto', vmin=0., vmax=1.)
    if save_plain_img:
        os.makedirs(img_path + '/imgs', exist_ok=True)
        #fig.colorbar(pos)
        plt.savefig(img_path + '/imgs/outputs_' + os.path.basename(img_path) + 's' + str(
            head - len(output_list_kpi)) + '_e' + str(
            head) + '__plain.png')

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

    os.makedirs(img_path + '/imgs', exist_ok=True)
    plt.savefig(img_path + '/imgs/outputs_' + os.path.basename(img_path) + 's' + str(head - len(output_list_kpi)) + '_e' + str(
        head) + postfix + '.png')
    plt.clf()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path containing the classifier output files for re-played traffic traces")
    parser.add_argument("--mode", choices=['pre-comp', 'inference'], default='pre-comp', help="Specify the type of file format we are trying to read.")
    parser.add_argument("--slicelen", choices=[4, 8, 16, 32, 64], type=int, default=32, help="Specify the slicelen to determine the classifier to load")
    parser.add_argument("--model", default='../model/model.32.cnn.pt', help="Specify the Torch model to load" )
    parser.add_argument("--Nclasses", default=4, help="Used to initialize the model")
    parser.add_argument("--chZeros", default=False, help="At test time, don't count the occurrences of ctrl class")
    args, _ = parser.parse_known_args()

    PATH = args.path
    if PATH[-1] == '/':  # due to use of os.basename
        PATH = PATH[:-1]

    check_zeros = args.chZeros

    if args.mode == 'pre-comp':
        slice_len = args.slicelen
        num_correct = 0
        output_list_kpi = []
        output_list_y = []
        head = 0

        pkl_list = glob.glob(os.path.join(PATH, 'class_output_*.pkl'))
        pkl_list.sort(key=natural_keys)

        for ix, p in enumerate(pkl_list):
            kpis, class_out = pickle.load(open(p, 'rb'))
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
                old_kpis = kpis.copy()
                for i in range(slice_len):
                    output_list_kpi.append(kpis[i,:])
                head = len(output_list_kpi)
            elif ix > 0:
                if np.all(kpis[0:slice_len-1, :] == old_kpis[1:slice_len, :]):
                    print('Kpis', ix, ' contiguous')
                    output_list_kpi.append(kpis[-1, :])
                    head += 1
                else:
                    print('Kpis', ix, 'NOT contiguous')
                    # first, let's plot everything until now and empty the output buffer
                    plot_trace_class(output_list_kpi, output_list_y, PATH, slice_len, head, save_plain_img=True)
                    pickle.dump(np.array(output_list_kpi), open(PATH+'/replay_kpis__'+ os.path.basename(PATH) + 's' + str(head - len(output_list_kpi)) + '_e' + str(
            head) +'.pkl', 'wb'))
                    # reset output lists
                    output_list_kpi = []
                    output_list_y = []

                    for i in range(slice_len):
                        output_list_kpi.append(kpis[i, :])
                    head += len(output_list_kpi)

                old_kpis = kpis.copy()

            output_list_y.append(class_out)

        # if there's data in the buffer
        if len(output_list_kpi) > 0:
            # let's print the accumulated KPI inputs and relative outputs
            imgout = np.array(output_list_kpi).T
            plot_trace_class(output_list_kpi, output_list_y, PATH, slice_len, head, save_plain_img=True)
            pickle.dump(np.array(output_list_kpi), open(
                PATH + '/replay_kpis__' + os.path.basename(PATH) + 's' + str(head - len(output_list_kpi)) + '_e' + str(
                    head) + '.pkl', 'wb'))
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

        if 'cnn' in args.model:
            global_model = ConvNN
        else:
            global_model = TransformerNN

        device = torch.device("cuda")

        pkl_list = glob.glob(os.path.join(PATH, 'replay*.pkl'))
        for ix, p in enumerate(pkl_list):
            kpis = pickle.load(open(p, 'rb'))
            model = global_model(classes=args.Nclasses, slice_len=args.slicelen, num_feats=kpis.shape[1])
            model.load_state_dict(torch.load(args.model, map_location='cuda:0')['model_state_dict'])
            model.to(device)
            model.eval()
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
            for t in range(kpis.shape[0]):
                if t + args.slicelen < kpis.shape[0]:
                    input_sample = kpis[t:t + args.slicelen]
                    input = torch.Tensor(input_sample[np.newaxis, :, :])
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

                    num_correct += 1 if (co == correct_class) else 0
                    num_samples += 1

            mypost = '_cnn_' if isinstance(model, ConvNN) else '_trans_'
            mypost += '_slice' + str(args.slicelen)
            mypost += '_chZero' if check_zeros else ''
            mypost += '_whitebg'
            plot_trace_class(kpis, output_list_y, PATH, args.slicelen, postfix=mypost, save_plain_img=True, hatchmap=None)
            mypost += '_hatch'
            plot_trace_class(kpis, output_list_y, PATH, args.slicelen, postfix=mypost, save_plain_img=True)
            print("Correct classification for traffic type (%): ", (num_correct / num_samples)*100., "num correct =", num_correct, ", num classifications =", num_samples)

            if check_zeros:
                unique, counts = np.unique(output_list_y, return_counts=True)
                count_class = dict(zip(unique, counts))
                print(count_class)
                if 3 in count_class.keys():
                    if num_heuristic_ctrl > 0:
                        print("Percent of verified ctrl (through heuristic): ", (num_verified_ctrl / num_heuristic_ctrl)*100., "num verified =", num_verified_ctrl, ", num heuristic matches =", num_heuristic_ctrl )
                    else:
                        print("No ctrl captured by the heuristic")
                else:
                    print("No ctrl captured by the heuristc")




