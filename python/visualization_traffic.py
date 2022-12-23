from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trial = 'Trial2'

embb_data = pd.read_csv("/home/mauro/Research/ORAN/traffic_gen/logs/"+trial+"/embb_clean.csv", sep=",")
mmtc_data = pd.read_csv("/home/mauro/Research/ORAN/traffic_gen/logs/"+trial+"/mmtc_clean.csv", sep=",")
urll_data = pd.read_csv("/home/mauro/Research/ORAN/traffic_gen/logs/" + trial + "/urll_clean.csv", sep=",")

if 'ul_rssi' in mmtc_data.columns:
    mmtc_data = mmtc_data.drop(['ul_rssi'], axis=1)

sampling_rate = 0.25    # secs
samps_1sec = int(1 / sampling_rate)


for ix, ds in enumerate([embb_data, mmtc_data, urll_data]):
    new_ds = []
    for i in range(ds.shape[0]):
        if i+samps_1sec < ds.shape[0]:
            new_ds.append(
                ds[i:i+samps_1sec].drop(['Timestamp'], axis=1)  # slice and remove the timestamp column
            )
    new_ds = np.array(new_ds)

    slice_len = new_ds.shape[1]
    feat_len = new_ds.shape[2]
    new_ds = np.reshape(new_ds, (new_ds.shape[0], slice_len * feat_len))

    if ix == 0:
        embb_data_flat = new_ds
    elif ix == 1:
        mmtc_data_flat = new_ds
    elif ix == 2:
        urll_data_flat = new_ds


allsamples_flat = np.concatenate((embb_data_flat, mmtc_data_flat, urll_data_flat), axis=0)

#km = KMeans(n_clusters=3)  # this method returns one centroid for every K-means feature
#km.fit(allsamples_flat)
#centers = km.cluster_centers_

TSNE_dim = 2
X_embedded = TSNE(n_components=TSNE_dim, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(allsamples_flat)

ds_info = {
    'embb': {'color': 'red', 'size': embb_data_flat.shape[0]},
    'mmtc': {'color': 'blue', 'size': mmtc_data_flat.shape[0]},
    'urllc': {'color': 'green', 'size': urll_data_flat.shape[0]}
}
start_samp = 0
n_samps = 0

if TSNE_dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for ix, ds in enumerate([embb_data_flat, mmtc_data_flat, urll_data_flat]):
        start_samp = n_samps
        n_samps += ds.shape[0]
        print("Plotting samples from", start_samp, "to", n_samps)
        ax.scatter(X_embedded[start_samp:n_samps, 0], X_embedded[start_samp:n_samps, 1], X_embedded[start_samp:n_samps, 2], marker='o', c=colors[ix])


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
else:
    for key, info in ds_info.items():
        start_samp = n_samps
        n_samps += info['size']
        print("Plotting samples from", start_samp, "to", n_samps)
        plt.scatter(X_embedded[start_samp:n_samps, 0], X_embedded[start_samp:n_samps, 1], marker='o', c=info['color'], label=key)
    plt.legend()
    plt.show()

    start_samp = 0
    n_samps = 0

    for key, info in ds_info.items():
        start_samp = n_samps
        n_samps += info['size']
        print("Plotting samples from", start_samp, "to", n_samps)
        plt.scatter(X_embedded[start_samp:n_samps, 0], X_embedded[start_samp:n_samps, 1], marker='o', c=info['color'], label=key)
        plt.legend()
        plt.show()




print('some stuff')