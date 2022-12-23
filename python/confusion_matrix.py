import glob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", required=True, help="Path containing Ray's logs with custom confusion matrix validation output.")
parser.add_argument("--figname", default='confusion_matrix.png', help="Name of confusion matrix image to be saved. NOTE: it will be saved in logdir")
args, _ = parser.parse_known_args()

#logdir = '/home/mauro/ray_results/traffic_class/TorchTrainer_2022-12-07_00-12-24/TorchTrainer_bc409_00000_0_2022-12-07_00-12-24'
#logdir = '/home/mauro/ray_results/TorchTrainer_2022-12-08_19-11-11/TorchTrainer_fceb0_00000_0_2022-12-08_19-11-11'
# 4 classes (first with timestep 4 and then with timestep 16)
#logdir = '/home/mauro/ray_results/traffic_class/Trial_12345/TorchTrainer_2022-12-09_19-11-46/TorchTrainer_3c04e_00000_0_2022-12-09_19-11-46'
#logdir = '/home/mauro/ray_results/TorchTrainer_2022-12-10_20-09-34/TorchTrainer_79973_00000_0_2022-12-10_20-09-34'
# 4 classes with cleaned up traces (i.e. without "silence")
# 16 step
# logdir = '/home/mauro/ray_results/traffic_class2/TorchTrainer_2022-12-13_21-19-35/TorchTrainer_c0bfc_00000_0_2022-12-13_21-19-35'
# 4 step
# logdir = '/home/mauro/ray_results/traffic_class2/TorchTrainer_2022-12-15_09-29-29/TorchTrainer_e23b1_00000_0_2022-12-15_09-29-29'
# 4 steps without Trial1
# logdir = '/home/mauro/ray_results/traffic_class2/TorchTrainer_2022-12-18_10-48-11/TorchTrainer_6031e_00000_0_2022-12-18_10-48-11'
# 4 steps including Trial6
# logdir = '/home/mauro/ray_results/traffic_class2/TorchTrainer_2022-12-18_20-48-53/TorchTrainer_4aee4_00000_0_2022-12-18_20-48-53'

logdir = args.logdir
figname = os.path.join(logdir, args.figname)

out = glob.glob(os.path.join(logdir, 'rank*'))
conf_mat = None
for rank_dir in out:
    cm = pickle.load(open(os.path.join(rank_dir, 'conf_matrix.last.pkl'), 'rb'))
    print(cm)
    for r in range(cm.shape[0]):
        sum_row = np.sum(cm[r,:])
        cm[r, :] = cm[r, :] / sum_row
    if conf_mat is None:
        conf_mat = cm
    else:
        conf_mat += cm


#plt.imshow(conf_mat)
#plt.colorbar()
axis_lbl = ['embb', 'mmtc', 'urll'] if conf_mat.shape[0] == 3 else ['embb', 'mmtc', 'urll', 'ctrl']
df_cm = pd.DataFrame(conf_mat/len(out), axis_lbl, axis_lbl)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

#plt.show()
plt.savefig(figname)
