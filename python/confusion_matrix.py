from glob import glob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", required=True, help="Path (can use wildcard *) containing Ray's logs with custom confusion matrix validation output.")
parser.add_argument("--figname", default='confusion_matrix.png', help="Name of confusion matrix image to be saved. NOTE: it will be saved in logdir")
args, _ = parser.parse_known_args()


logdir = args.logdir
dirs = glob(logdir)

exp_accuracies = []

for d in sorted(dirs):
    figname = os.path.join(d, args.figname)

    out = glob(os.path.join(d, '*/rank*'))
    conf_mat = None
    tot_samples = []
    correct_samples = []

    for rank_dir in out:
        cm = pickle.load(open(os.path.join(rank_dir, 'conf_matrix.last.pkl'), 'rb'))
        #print(cm)

        tot_s = np.sum(cm)
        correct_s = 0
        for r in range(cm.shape[0]):
            correct_s += cm[r, r]
        tot_samples.append(tot_s)
        correct_samples.append(correct_s)

        for r in range(cm.shape[0]):    # for each row in the confusion matrix
            sum_row = np.sum(cm[r,:])
            cm[r, :] = cm[r, :] / sum_row   # compute in percentage
            # also compute accuracy for each row

        if conf_mat is None:
            conf_mat = cm
        else:
            conf_mat += cm

    if True: #len(dirs) == 1:
        plt.clf()
        axis_lbl = ['embb', 'mmtc', 'urll'] if conf_mat.shape[0] == 3 else ['embb', 'mmtc', 'urll', 'ctrl']
        df_cm = pd.DataFrame(conf_mat/len(out), axis_lbl, axis_lbl)
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

        #plt.show()
        plt.savefig(figname)

    # compute average accuracy from all workers
    workers_accuracy = [c/t for c,t in zip(correct_samples, tot_samples)]
    exp_accuracies.append(np.mean(workers_accuracy)*100)

if len(dirs) > 1:
    plt.clf()
    plt.rcParams.update({"figure.figsize" : (6,5)})
    # plt.plot(exp_accuracies, 's--', label='4 class')
    # plt.xticks(np.arange(len(exp_accuracies)), ['4', '8', '16', '32'])
    sn.set(font_scale=1.2)  # for label size
    plt.plot([4,8,16,32], exp_accuracies, 's--', label='4 class')
    plt.xticks([4,8,16,32], ['4', '8', '16', '32'])
    plt.xlabel('Slice size')
    plt.ylabel('Accuracy (%)')
    #plt.legend()
    plt.grid(color='gray', linestyle='--')
    plt.savefig('4class_accuracy.png')





