import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default='/home/jerry/Downloads/raw_josh/', help="Path of main directory containing different classification outputs (the folder containing folders of outputs)")
parser.add_argument("--savedir", default='/home/jerry/Downloads/raw_josh/', help="Path where traffic and interefence confusion matrices will be saved")
parser.add_argument("--ctrl", default=False, help="Visualize ctrl as a true label, traffic classification only")
args, _ = parser.parse_known_args()

def show_all_files_in_directory(input_path,extension):
    #'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
                files_list.append(os.path.join(path, file))
    return files_list

def cm_percent(cs):
    sum_row = np.sum(cs)
    if sum_row == 0:
        return cs
    x = np.divide(cs,sum_row)  # compute in percentage
    return x

logdir = args.logdir
savedir = args.savedir
ctrl = args.ctrl
files = show_all_files_in_directory(logdir,'.pkl')

embb_tot = np.array([0,0,0,0])
mmtc_tot = np.array([0,0,0,0])
urll_tot = np.array([0,0,0,0])

embb_i_tot = np.array([0,0,0,0]) # For folders with interference '_i'
mmtc_i_tot = np.array([0,0,0,0])
urll_i_tot = np.array([0,0,0,0])

none_tot = np.array([0,0,0])
interfere_tot = np.array([0,0,0])

if ctrl != True:
    for i in tqdm(sorted(files)):
        embb_cnt = 0
        mmtc_cnt = 0
        urll_cnt = 0
        ctrl_cnt = 0

        embb_i_cnt = 0
        mmtc_i_cnt = 0
        urll_i_cnt = 0
        ctrl_i_cnt = 0

        none_cnt = 0
        interfere_cnt = 0
        ctrl_interfere_cnt = 0

        if 'class_' in i: # Currently splits traffic from traffic with interference results
            if '_i/' in i:
                current_cnt=np.array([0,0,0,0])
                cm = pickle.load(open(i, 'rb'))
                pred=int(cm[1])
                if pred==0:
                    embb_i_cnt+=1
                elif pred==1:
                    mmtc_i_cnt+=1
                elif pred==2:
                    urll_i_cnt+=1
                elif pred==3:
                    ctrl_i_cnt+=1
                else:
                    print('Invalid prediction')
                current_cnt=np.array([embb_i_cnt,mmtc_i_cnt,urll_i_cnt,ctrl_i_cnt])
                #print(i)
                if 'embb' in i:
                    embb_i_tot+=current_cnt
                elif 'mmtc' in i:
                    mmtc_i_tot+=current_cnt
                elif 'urll' in i:
                    urll_i_tot+=current_cnt
                else: # not reading mixed
                    print('Single true label folders only')
            else:
                current_cnt=np.array([0,0,0,0])
                cm = pickle.load(open(i, 'rb'))
                pred=int(cm[1])
                if pred==0:
                    embb_cnt+=1
                elif pred==1:
                    mmtc_cnt+=1
                elif pred==2:
                    urll_cnt+=1
                elif pred==3:
                    ctrl_cnt+=1
                else:
                    print('Invalid prediction')
                current_cnt=np.array([embb_cnt,mmtc_cnt,urll_cnt,ctrl_cnt])
                #print(i)
                if 'embb' in i:
                    embb_tot+=current_cnt
                elif 'mmtc' in i:
                    mmtc_tot+=current_cnt
                elif 'urll' in i:
                    urll_tot+=current_cnt
                else: # not reading mixed
                    print('Single true label folders only')

        elif 'interference_' in i:
            current_cnt=np.array([0,0,0,0])
            cm = pickle.load(open(i, 'rb'))
            pred=int(cm[1])
            if pred==0:
                none_cnt+=1
            elif pred==1:
                interfere_cnt+=1
            elif pred==2:
                ctrl_interfere_cnt+=1
            else:
                print('Invalid prediction')
            current_cnt=[none_cnt,interfere_cnt,ctrl_interfere_cnt]
            if '_i/' in i:
                interfere_tot+=current_cnt
            else:
                none_tot+=current_cnt

        else:
            print('Invalid file: ',i)

    embb_percent=cm_percent(embb_tot)
    mmtc_percent=cm_percent(mmtc_tot)
    urll_percent=cm_percent(urll_tot)

    embb_i_percent=cm_percent(embb_i_tot)
    mmtc_i_percent=cm_percent(mmtc_i_tot)
    urll_i_percent=cm_percent(urll_i_tot)

    none_percent=cm_percent(none_tot)
    interfere_percent=cm_percent(interfere_tot)

    traffic_mat=np.vstack((embb_percent,mmtc_percent,urll_percent))
    plt.clf()
    traffic_cm = pd.DataFrame(traffic_mat, ['eMBB', 'mMTC', 'URLLC'], ['eMBB', 'mMTC', 'URLLC', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(traffic_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Traffic Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'traffic_results.png')

    traffic_i_mat=np.vstack((embb_i_percent,mmtc_i_percent,urll_i_percent))
    plt.clf()
    traffic_i_cm = pd.DataFrame(traffic_i_mat, ['eMBB', 'mMTC', 'URLLC'], ['eMBB', 'mMTC', 'URLLC', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(traffic_i_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Traffic w/ Interference Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'traffic_i_results.png')

    interference_mat=np.vstack((none_percent,interfere_percent))
    plt.clf()
    interference_axis_lbl = ['None', 'Interference', 'ctrl']
    interference_cm = pd.DataFrame(interference_mat, ['None', 'Interference'], ['None', 'Interference', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(interference_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Interference Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'interference_results.png')

else:
    ctrl_tot = np.array([0,0,0,0])
    ctrl_i_tot = np.array([0,0,0,0])
    for i in tqdm(sorted(files)):
        embb_cnt = 0
        mmtc_cnt = 0
        urll_cnt = 0
        ctrl_cnt = 0

        embb_i_cnt = 0
        mmtc_i_cnt = 0
        urll_i_cnt = 0
        ctrl_i_cnt = 0

        none_cnt = 0
        interfere_cnt = 0
        ctrl_interfere_cnt = 0

        if 'class_' in i: # Currently splits traffic from traffic with interference results
            if '_i/' in i:
                current_cnt=np.array([0,0,0,0])
                cm = pickle.load(open(i, 'rb'))
                pred=int(cm[1])
                if pred==0:
                    embb_i_cnt+=1
                elif pred==1:
                    mmtc_i_cnt+=1
                elif pred==2:
                    urll_i_cnt+=1
                elif pred==3:
                    ctrl_i_cnt+=1
                else:
                    print('Invalid prediction')
                current_cnt=np.array([embb_i_cnt,mmtc_i_cnt,urll_i_cnt,ctrl_i_cnt])
                #print(i)
                if 'embb' in i:
                    embb_i_tot+=current_cnt
                elif 'mmtc' in i:
                    mmtc_i_tot+=current_cnt
                elif 'urll' in i:
                    urll_i_tot+=current_cnt
                elif 'ctrl' in i or 'null' in i:
                    ctrl_i_tot+=current_cnt
                else: # not reading mixed
                    print('Single true label folders only')
            else:
                current_cnt=np.array([0,0,0,0])
                cm = pickle.load(open(i, 'rb'))
                pred=int(cm[1])
                if pred==0:
                    embb_cnt+=1
                elif pred==1:
                    mmtc_cnt+=1
                elif pred==2:
                    urll_cnt+=1
                elif pred==3:
                    ctrl_cnt+=1
                else:
                    print('Invalid prediction')
                current_cnt=np.array([embb_cnt,mmtc_cnt,urll_cnt,ctrl_cnt])
                #print(i)
                if 'embb' in i:
                    embb_tot+=current_cnt
                elif 'mmtc' in i:
                    mmtc_tot+=current_cnt
                elif 'urll' in i:
                    urll_tot+=current_cnt
                elif 'ctrl' in i or 'null' in i:
                    ctrl_tot+=current_cnt
                else: # not reading mixed
                    print('Single true label folders only')

        elif 'interference_' in i:
            current_cnt=np.array([0,0,0,0])
            cm = pickle.load(open(i, 'rb'))
            pred=int(cm[1])
            if pred==0:
                none_cnt+=1
            elif pred==1:
                interfere_cnt+=1
            elif pred==2:
                ctrl_interfere_cnt+=1
            else:
                print('Invalid prediction')
            current_cnt=[none_cnt,interfere_cnt,ctrl_interfere_cnt]
            if '_i/' in i:
                interfere_tot+=current_cnt
            else:
                none_tot+=current_cnt

        else:
            print('Invalid file: ',i)

    embb_percent=cm_percent(embb_tot)
    mmtc_percent=cm_percent(mmtc_tot)
    urll_percent=cm_percent(urll_tot)
    ctrl_percent=cm_percent(ctrl_tot)

    embb_i_percent=cm_percent(embb_i_tot)
    mmtc_i_percent=cm_percent(mmtc_i_tot)
    urll_i_percent=cm_percent(urll_i_tot)
    ctrl_i_percent=cm_percent(ctrl_i_tot)

    none_percent=cm_percent(none_tot)
    interfere_percent=cm_percent(interfere_tot)

    traffic_mat=np.vstack((embb_percent,mmtc_percent,urll_percent, ctrl_percent))
    plt.clf()
    traffic_cm = pd.DataFrame(traffic_mat, ['eMBB', 'mMTC', 'URLLC', 'ctrl'], ['eMBB', 'mMTC', 'URLLC', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(traffic_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Traffic Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'traffic_results.png')

    traffic_i_mat=np.vstack((embb_i_percent,mmtc_i_percent,urll_i_percent, ctrl_i_percent))
    plt.clf()
    traffic_i_cm = pd.DataFrame(traffic_i_mat, ['eMBB', 'mMTC', 'URLLC', 'ctrl'], ['eMBB', 'mMTC', 'URLLC', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(traffic_i_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Traffic w/ Interference Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'traffic_i_results.png')

    interference_mat=np.vstack((none_percent,interfere_percent))
    plt.clf()
    interference_axis_lbl = ['None', 'Interference', 'ctrl']
    interference_cm = pd.DataFrame(interference_mat, ['None', 'Interference'], ['None', 'Interference', 'ctrl'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(interference_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('Interference Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(savedir+'interference_results.png')
