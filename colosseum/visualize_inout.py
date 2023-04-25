import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path containing the classifier output files for re-played traffic traces")
args, _ = parser.parse_known_args()

import numpy
import pickle
import glob
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import sys
sys.path.append(proj_root_dir)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PATH=args.path
if PATH[-1] == '/': # due to use of os.basename
    PATH = PATH[:-1]

pkl_list = glob.glob(os.path.join(PATH, 'class_output_*.pkl'))
classmap = {'embb': 0, 'mmtc': 1, 'urll': 2, 'ctrl': 3}
colormap = {0: 'y', 1: 'r', 2: 'g', 3: 'b'}

slice_len = 8
num_correct = 0
output_list_kpi = []
output_list_y = []
head = 0

import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

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
            imgout = np.array(output_list_kpi).T
            # Create figure and axes
            fig, ax = plt.subplots()
            # Display the image
            ax.imshow(imgout, vmin=0., vmax=1.)
            for ix, label in enumerate(output_list_y):
                lbl = label.numpy()[0]
                # Create a Rectangle patch
                #rect = patches.Rectangle((ix, 0), slice_len, kpis.shape[1]-1, linewidth=0.5, edgecolor=colormap[lbl], facecolor='none')
                rect = patches.Rectangle((ix, 0), slice_len, kpis.shape[1]-1, linewidth=1, edgecolor=colormap[lbl], facecolor=colormap[lbl], alpha=0.15)
                # Add the patch to the Axes
                ax.add_patch(rect)
            os.makedirs(PATH+'/imgs', exist_ok=True)
            plt.savefig(PATH+'/imgs/outputs_s'+os.path.basename(PATH)+str(head-len(output_list_kpi))+'_e'+str(head)+'.png')
            plt.clf()
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
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(50, 5))
    # Display the image
    ax.imshow(imgout, extent=[0,len(output_list_kpi),0,kpis.shape[1]], aspect='auto', vmin=0., vmax=1.)
    for ix, label in enumerate(output_list_y):
        lbl = label.numpy()[0]
        # Create a Rectangle patch
        #rect = patches.Rectangle((ix, 0), slice_len, kpis.shape[1]-1, linewidth=0.5, edgecolor=colormap[lbl], facecolor='none')
        rect = patches.Rectangle((ix, 0), slice_len, kpis.shape[1]-1, linewidth=1, edgecolor=colormap[lbl], facecolor=colormap[lbl], alpha=0.15)
        # Add the patch to the Axes
        ax.add_patch(rect)
    os.makedirs(PATH+'/imgs', exist_ok=True)
    plt.savefig(PATH+'/imgs/outputs_'+os.path.basename(PATH)+'s'+str(head-len(output_list_kpi))+'_e'+str(head)+'.pdf')
    plt.clf()

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


