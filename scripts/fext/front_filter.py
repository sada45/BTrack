import numpy as np
import matplotlib.pyplot as plt 
import utils
import feature_extractor as fe
from sklearn.cluster import MeanShift
import time
import config

# 1-d version of mahalanobis distance
def mahalanobis_distance(x, clusters):
    d = np.abs(x[:, np.newaxis] - clusters[:, 0])
    md = d / clusters[:, 1]
    return md

def save_csv(f_path, cluster_id, node_id, cfo, distance):
    with open(f_path, 'w') as f:
        for i in range(len(cluster_id)):
            line = "{}, {}, {}, {}, {}\n".format(int(cluster_id[i]), int(node_id[i]), format(cfo[i], '.2f'), format(distance[i, 0], '.2f'), format(distance[i, 1], '.2f'))
            f.write(line)
        f.flush()

def cluster_cfo(X):
    meanshift = MeanShift(bandwidth=20, cluster_all=False)  # allow some orphans
    cluster = meanshift.fit_predict(X.reshape([-1, 1]))
    cluster = cluster.astype(np.int32)
    cluster_num = int(np.max(cluster) + 1)
    cluster_info = []
    for i in range(cluster_num):
        cfos = X[np.where(cluster==i)]
        cluster_info.append(np.array([np.mean(cfos), np.std(cfos)]))
    cluster_info = np.array(cluster_info)
    distance = mahalanobis_distance(X, cluster_info)
    # min_distance_idx = np.argmin(distance, axis=1)
    # min_distance = np.min(distance, axis=1)
    # save_csv("./output/cluster.csv", cluster, nodeid, X, distance)
    return cluster_info, distance

all_dev_name = ["cc2650"]
dev_name = all_dev_name[:3]
st_clusters = []
cfo_clusters = []
for d in dev_name:
    data = np.load("./data/" + d + "_cluster_info.npz")
    st_clusters.append(data["st_info"])
    ci = data["cluster_info"]
    # Set the cfo limits no larger than 20KHz 
    # for i in range(ci.shape[0]):
    #     if ci[i, 1] * 10 > 20:
    #         ci[i, 1] = 2
    cfo_clusters.append(ci)
st_clusters = np.array(st_clusters)
print(st_clusters, cfo_clusters)


def st_filter(sts, thres=3):
    distance = mahalanobis_distance(sts, st_clusters)
    min_idx = np.argmin(distance, axis=1)
    min_val = np.min(distance, axis=1)
    valid_idx = np.where(min_val<=thres)[0]  # Remove the alien devices 
    min_idx = min_idx[valid_idx]
    # min_val = min_val[valid_idx]
    return distance[valid_idx, :], valid_idx

def cfo_filter(cfos, st_distance, thres=3, sts_thres=2):
    valid_idx = []
    st_cluster_idxs = []

    for i in range(cfos.shape[0]):
        flag = False
        corr_idx = np.zeros(st_distance.shape[1], dtype=np.uint8)
        for j in range(st_distance.shape[1]):
            if st_distance[i, j] <= sts_thres:
                cfo_distance = mahalanobis_distance(cfos[i:i+1], cfo_clusters[j])[0]
                if np.min(cfo_distance) <= thres:
                    corr_idx[j] = 1
                    flag = True
            if flag:
                st_cluster_idxs.append(corr_idx)
                valid_idx.append(i)
    return np.array(st_cluster_idxs), valid_idx

def front_filter(sts, cfos, thres=3):
    st_disatance, st_valid_idx = st_filter(sts, 2)
    st_cluster_idxs, cfo_valid_idx = cfo_filter(cfos[st_valid_idx], st_disatance, thres)
    total_valid_idx = st_valid_idx[cfo_valid_idx]
    return total_valid_idx, st_cluster_idxs

feat_lens = {"nrf": 600,
             "cc2650":800,
             "da":800}

data = np.load("/data/blueprint/raw_data/no_signal/chan=37.npz")
raw_data = data["arr_0"]
filter_signal = utils.filter(raw_data)
comp_phase_diff = utils.get_phase_diff(filter_signal)
preamble_idx, _ = utils.find_preamble(comp_phase_diff, native=True, compare_num=32)
phase = utils.get_phase(filter_signal)
cfo, _, _ = utils.get_cfo(phase, preamble_idx)
ents = fe.get_ramp_seg_cfo(filter_signal, preamble_idx, cfo, raw_data, native=True, derivate_thres=5e-3, hight_diff_thres=0.01, feat_len=feat_lens[all_dev_name[0]], time_stamp=None, preamble_bias=0)
st = ents[2] / config.sample_rate
cfo = cfo[ents[4]] / 1000
# for i in range(cfo.shape[0]):
#     print(i, cfo[i], st[i])
start_time = time.time()
valid_idx, cluster_idx = front_filter(st, cfo)
end_time = time.time()
print(len(valid_idx)/raw_data.shape[0])
print((end_time - start_time) / len(st))

# [[1.50327194e-05 1.67004342e-06]] [array([[-9.61051805,  2.        ]])]