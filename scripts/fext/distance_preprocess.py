import numpy as np
import matplotlib.pyplot as plt 
import feature_extractor as fe
import basic_feature_extractor as bfe
from sklearn.cluster import MeanShift


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

all_dev_name = ["nrf", "esp", "cc2650"]
dev_name = all_dev_name[:3]
st_clusters = []
cfo_clusters = []



def st_filter(sts, thres=3):
    distance = mahalanobis_distance(sts, st_clusters)
    min_idx = np.argmin(distance, axis=1)
    min_val = np.min(distance, axis=1)
    valid_idx = np.where(min_val<=thres)[0]  # Remove the alien devices 
    min_idx = min_idx[valid_idx]
    # min_val = min_val[valid_idx]
    return distance[valid_idx, :], valid_idx

def cfo_filter(cfos, st_distance, thres=3):
    valid_idx = []
    st_cluster_idxs = []
    for i in range(cfos.shape[0]):
        flag = False
        corr_idx = np.zeros(st_distance.shape[1], dtype=np.uint8)
        for j in range(st_distance.shape[1]):
            if st_distance[i, j] <= thres:
                cfo_distance = mahalanobis_distance(cfos[i:i+1], cfo_clusters[j])[0]
                if np.min(cfo_distance) <= thres:
                    corr_idx[j] = 1
                    flag = True
        if flag > 0:
            st_cluster_idxs.append(corr_idx)
            valid_idx.append(i)
    return np.array(st_cluster_idxs), valid_idx

def front_filter(sts, cfos, thres=3):
    st_disatance, st_valid_idx = st_filter(sts, thres)
    st_cluster_idxs, cfo_valid_idx = cfo_filter(cfos[st_valid_idx], st_disatance, thres)
    total_valid_idx = st_valid_idx[cfo_valid_idx]
    return total_valid_idx, st_cluster_idxs




node_id = {"nrf":[i for i in range(1, 13)], 
        "esp":[i for i in range(1, 9)],
        "cc2650": [i for i in range(2, 7)],
        "da":[i for i in range(1, 8)]}

pre_cfos = {"nrf": bfe.cfo_list, 
            "esp": None,
            "cc2650": None, 
            "da": None}
is_natives = {"nrf": False, 
            "esp": True,
            "cc2650": True, 
            "da": True,}
feat_lens = {"nrf": 600,
             "esp":600,
             "cc2650":800,
             "da":800}
test_num = {"nrf":2,
            "cc2650":1,
            "da":2}
scale = {"nrf":1,
         "cc2650":1,
         "da":6,
         "esp":1}
feat_lens = {"nrf": 600,
             "esp":400,
             "cc2650":800,
             "da":800}
# This idx means the idx in the node_id
test_id = {"nrf": [0, 1],
            "cc2650":[2],
            "da":[1, 2]}

def generate_front_filter(dev_name):
    f = bfe.get_features(path.format(dev_name, 1, "{}", "{}"), [i for i in range(0, 4)], node_id[dev_name], -1, native=is_natives[dev_name], bias=0, pre_cfo=pre_cfos[dev_name], feat_len=feat_lens[dev_name])
    cluster_info, _ = cluster_cfo(f[0])
    print(cluster_info)
    st_info = np.array([np.mean(f[1]), np.std(f[1])])
    np.savez("./data/" + dev_name + "_cluster_info.npz", st_info=st_info, cluster_info=cluster_info)
    # for i in range(10):
    #     plt.figure()
    #     plt.plot(f[4][i, 0, :])
    #     plt.savefig("./fe/" +str(i) + ".pdf")
    #     plt.close()

# generate_front_filter("da")
# path = "/data/blueprint/raw_data/distance/{}/new/d={}m_ttt_{}_nodeid_{}_chan=37.npz"
# path = "/data/blueprint/raw_data/distance/{}/d={}m_ttt_{}_nodeid_{}_chan=37.npz"
path = "/data/blueprint/raw_data/tracking/{}/d={}m_east_ttt_{}_nodeid_{}_chan=37.npz"
# for l in [400, 500, 600, 700, 800]:
#     feat_lens["nrf"] = l
if __name__ == "__main__":
    for key in ["nrf"]:
        st_clusters = []
        cfo_clusters = []
        # generate_front_filter(key)
        """init the st_cluster"""
        # for d in [key]:
        #     data = np.load("./data/" + d + "_cluster_info.npz")
        #     st_clusters.append(data["st_info"])
        #     cfo_clusters.append(data["cluster_info"])
        #     print(d, data["st_info"], data["cluster_info"])
        #     st_info = data["st_info"]
            # print(d, 600 / 20e6 - st_info[0], st_info[1])
        # st_clusters = np.array(st_clusters)

        # 1, 3, 5, 7, 9, 11
        for dis in [1, 3, 5, 7, 9, 11]:
            # if key == "nrf" and dis == 7:
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(4, 8)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            # elif key == "nrf" and dis == 3:
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(20, 24)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            # elif key == "nrf" and dis == 5:
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(16, 20)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])

            # elif key == "da" and (dis == 3):
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(12, 16)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            # elif key == "da" and dis==7:
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(12, 16)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            # elif key == "da" and dis==9:
            #     f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(, 16)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            
            # else:
            f = bfe.get_features(path.format(key, dis, "{}", "{}"), [i for i in range(0, 4)], node_id[key], -1, native=is_natives[key], bias=0, pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            if dis == 1:
                cluster_info, _ = cluster_cfo(f[0])
                print(cluster_info)
                st_info = np.array([np.mean(f[1]), np.std(f[1])])
                np.savez("./data/" + key + "_cluster_info.npz", st_info=st_info, cluster_info=cluster_info)
                st_clusters.append(st_info)
                cfo_clusters.append(cluster_info)
                st_clusters = np.array(st_clusters)
            
            ramp_seg = f[4] / scale[key]
        
            labels = f[5]
            sts = f[1]
            _, valid_idx = st_filter(sts, 2)
            print(dis, len(valid_idx))
            np.savez("./data/nn_data/{}_{}m.npz".format(key, dis), data=ramp_seg[valid_idx, :], label=labels[valid_idx, :], cfo=f[0][valid_idx])
        
        sites = ["conference", "terrace", "corridor"]
        site_path = "/data/blueprint/raw_data/other_place/{}/{}/d=1m_ttt_{}_nodeid_{}_chan=37.npz"
        for s in sites:
            f = bfe.get_features(site_path.format(s, key, "{}", "{}"), [i for i in range(0, 4)], test_id[key], -1, native=is_natives[key], bias=0, origin_id=node_id[key], pre_cfo=pre_cfos[key], feat_len=feat_lens[key])
            ramp_seg = f[4] / scale[key]
            labels = f[5]
            sts = f[1]
            _, valid_idx = st_filter(sts, 2)
            print(s, len(valid_idx))
            np.savez("./data/nn_data/{}_{}.npz".format(key, s), data=ramp_seg[valid_idx, :], label=labels[valid_idx, :], cfo=f[0][valid_idx])




"""预实验，证明一些动态性指标可以用来判别不同设备"""
# for nid in [3, 4, 5]:
#     f = bfe.get_features("/data/blueprint/raw_data/nrf52840_raw/d=0_{}_nodeid_{}_chan=37.npz", [3], [nid], -1, native=False, bias=0, pre_cfo=None, feat_len=600)
#     np.savetxt("./output/nrf_ramp_{}.csv".format(nid), f[4][21, 0, :], delimiter=",")
#     ramp_seg = f[4]
#     time_stamp = f[8]
#     cfo = f[0]
#     _, _, _, mf = fe.manual_feature_ext(ramp_seg[:, 0, :], time_stamp, cfo)
#     min_val = mf[3]
#     stable_val = mf[4]
#     min_idx = (mf[2]-200) / 20e6
#     tw = mf[0] / 20e6
#     os = mf[1]
#     print(nid, "=====")
#     print("min_val", np.mean(min_val), np.std(min_val))
#     print("min_idx", np.mean(min_idx), np.std(min_idx))
#     print("trough width", np.mean(tw), np.std(tw))
#     print("overshot", np.mean(os), np.std(os))
#     print("stable val", np.mean(stable_val), np.std(stable_val))
#     # for i in range(50):
#     #     plt.figure()
#     #     plt.plot(ramp_seg[i, 0, :])
#     #     plt.savefig("./fe/t_" + str(i) + ".pdf")
#     #     plt.close()
