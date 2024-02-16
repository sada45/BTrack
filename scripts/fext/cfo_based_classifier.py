import numpy as np

def mahalanobis_distance(x, clusters):
    d = np.abs(x[:, np.newaxis] - clusters[:, 0])
    md = d / clusters[:, 1]
    return md


def train_process(f_name, train_idx, train_ratio=0.8):
    # print(train_idxa)
    raw_data = np.load(f_name)
    cfo_cluster = []
    label = np.argmax(raw_data["label"], axis=1)
    cfo = raw_data["cfo"]

    for tid in train_idx:
        idx = np.where(label==tid)[0]
        train_num = int(len(idx) * train_ratio)
        idx = idx[:train_num]
        t_cfo = cfo[idx]
        cfo_cluster.append(np.array([np.mean(t_cfo), np.std(t_cfo)]))
    return np.array(cfo_cluster)

def classify(cfos, cfo_cluster, target_nodeid, thres=3):
    distance = mahalanobis_distance(cfos, cfo_cluster)
    min_dis = np.min(distance, axis=1)
    min_idx = np.argmin(distance, axis=1)
    predict_label = np.zeros(len(cfos))
    # predict_label[np.where(min_dis>=thres)] = -1  # The others type
    for i in range(len(cfos)):
        if thres > 0 and min_dis[i] >= thres:
            predict_label[i] = -1  # others
            continue
        if min_idx[i] < len(target_nodeid):
            predict_label[i] = target_nodeid[min_idx[i]]
        else:
            predict_label[i] = -1  # others
    return predict_label

def get_dataset(f_name, target_nodeid, test=False, train_idx=None, train_ratio=0.8, alien=None):
    raw_data = np.load(f_name)
    cfo = raw_data["cfo"]
    label = raw_data["label"]
    type = np.argmax(label, axis=1)
    for i in range(len(type)):
        if type[i] not in target_nodeid:
            if alien is not None and type[i] in alien:
                type[i] = -2  #alien
            else:
                type[i] = -1  # The others
    
    valid_idx = []
    if test:
        for i in range(label.shape[1]):
            if i in train_idx:
                type_idx = np.where(label[:, i]==1)[0]
                train_num = int(len(type_idx) * train_ratio)
                test_num = len(type_idx) - train_num
                valid_idx.append(type_idx[train_num:train_num+test_num])
            else:
                type_idx = np.where(label[:, i]==1)[0]
                valid_idx.append(type_idx)    
        valid_idx = np.concatenate(valid_idx)
        return cfo[valid_idx], type[valid_idx]
    else:
        return cfo, type


def cfo_matching_rates():
    dev_name = "nrf"
    dataset_path = "/data/blueprint/data/nn_data/unique_{}.npz".format(dev_name)
    cfo_cluster = train_process(dataset_path, [i for i in range(12)], train_ratio=0.8)
    print(cfo_cluster)
    cfos, types = get_dataset(dataset_path, [i for i in range(12)], train_ratio=0)
    distance = mahalanobis_distance(cfos, cfo_cluster)
    matching_rates = []
    for i in range(12):
        all_type_idx = np.where(types==i)[0]
        poss = np.zeros((len(all_type_idx), 12), dtype=np.float32)
        d = -distance[all_type_idx, :]
        exp = np.exp(d)
        sum_exp = np.sum(exp, axis=1)[:, np.newaxis]
        poss = exp / sum_exp
        poss = np.sum(poss, axis=0) / len(all_type_idx)
        matching_rates.append(poss)
    mr = np.array(matching_rates)
    # print(np.sum(mr, axis=1))
    # print(mr)
    np.savetxt("./output/cfo_matching_rates.csv", mr, delimiter=",")

# cfo_matching_rates()






node_id = {"nrf":[i for i in range(12)], 
        "esp":[i for i in range(1, 9)],
        "cc2650": [i for i in range(5)],
        "da":[i for i in range(7)]}


target_ids = {"nrf": [0, 1],
              "cc2650": [2],
              "da": [1, 2]}
other_ids = {"nrf": [2, 3, 4, 5, 10, 11], 
             "cc2650": [0, 1, 3],
              "da": [0, 3, 4, 5]}
alien_ids = {"nrf": [6, 7, 8, 9], 
              "cc2650": [4],
              "da": [6]}


path = "/data/blueprint/raw_data/distance/{}/d={}m_ttt_{}_nodeid_{}_chan=37.npz"
dataset_path = "/data/blueprint/data/nn_data/{}_{}m.npz"
temp_dataset_path = "/data/blueprint/data/nn_data/temp_{}_{}c.npz"
site_dataset_path = "/data/blueprint/data/nn_data/{}_{}.npz"


dev_names = ["da"]

if __name__ == "__main__":
    for name in dev_names:
        """
        Actually, CFO-based method can run without the "others" group.
        It can just train with the "target", the accuracy of the "target" set is higher since there is no overlaps with "others" group so the accuracy is the highest.
        However, this method can have a high False Positive Rate, since the Mahalanobis distance of 2 or 3 can actually cover lots of different chip, it is not practical and unfair to compare with our method
        Even with this setting, the accuracy of "target" significantly drops with temperature since the CFO change significantly and the distance between the average CFO exceeds the distance threshold

        nrf: 
            0.9748743718592965, 0.8918099089989889, 0.6555217831813577, 0, 0   
            others:0.6551521099116782 (higher than other models since it has two revision with two totally different CFO ranges)
        cc2650: 
            0.9705882352941176, 0.9469548133595285, 0.8952569169960475, 0.7254901960784313, 0.3175542406311637
            others:0.2581479565442318
        da: 
            0.6010362694300518, 0.46239837398373984, 0.4837355718782791, 0.1851040525739321, 0.1555793991416309
            others:0.35810276679841896
        """
        # thres = 2
        # train_ids = target_ids[name]

        thres = 3
        train_ids = target_ids[name] + other_ids[name]

        """
        The code for unique identification
        """
        # thres = 0
        # train_ids = target_ids[name] + other_ids[name] + alien_ids[name]

        print(train_ids)
        cfo_cluster = train_process(temp_dataset_path.format(name, 16), train_ids)
        print(cfo_cluster)
        for temp in [16, 20, 25, 30, 35]:
            cfo_cluster1 = train_process(temp_dataset_path.format(name, temp), target_ids[name], train_ratio=1)
            print(temp, cfo_cluster1)

            if temp == 16:
                cfos, types = get_dataset(temp_dataset_path.format(name, temp), target_ids[name], True, train_ids)
                # The code for unique identification
                # cfos, types = get_dataset(temp_dataset_path.format(name, temp), train_ids, True, train_ids)
            else:
                cfos, types = get_dataset(temp_dataset_path.format(name, temp), target_ids[name])
                # The code for unique identification
                # cfos, types = get_dataset(temp_dataset_path.format(name, temp), train_ids)
            predict_type = classify(cfos, cfo_cluster, target_ids[name], thres)
            # The code for unique identification
            # predict_type = classify(cfos, cfo_cluster, train_ids, thres)
            others_idx = np.where(types==-1)[0]
            alien_idx = np.where(types==-2)[0]
            others_correct_num = len(np.where(predict_type[others_idx]==-1)[0])
            alien_correct_num = len(np.where(predict_type[alien_idx]==-1)[0])
            if len(others_idx) > 0:
                others_acc = others_correct_num / len(others_idx)
            else:
                others_acc = -1
            if len(alien_idx) > 0:
                alien_acc = alien_correct_num  / len(alien_idx)
            else:
                alien_acc = -1
            if (len(others_idx) + len(alien_idx)) > 1:
                total_other_acc = (others_correct_num + alien_correct_num) / (len(others_idx) + len(alien_idx))
            else:
                total_other_acc = -1
            target_idx = np.where((types!=-1) & (types!=-2))[0]
            target_correct_num = len(np.where(predict_type[target_idx]==types[target_idx])[0])
            target_acc = target_correct_num / len(target_idx)
            # t = np.zeros(2)
            # tn = np.zeros(2)
            # for i in range(len(target_idx)):
            #     tn[types[target_idx[i]]] += 1
            #     if predict_type[target_idx[i]] == types[target_idx[i]]:
            #         t[types[target_idx[i]]] += 1
            # print(t / tn)
            print(name, "temp:{}, target:{}, others:{}, alien:{}, total_others:{}".format(temp, target_acc, others_acc, alien_acc, total_other_acc))







    


