import numpy as np
import feature_extractor as fe
import matplotlib.pyplot as plt 
import utils
import config
import time

cfo_list = np.array([108090.01250352601, 113508.6613375386, 110338.2471902627, -12720.419896746136, 113164.02264103664, -14845.531992728753, 8933.615425365566, -18704.24309504716, -10890.547783049955, 112777.59449315938, 106871.18719910384, -7755.785061007704])


def get_features(prefix, sample_ids, ids, num, native, bias, origin_id=None, pre_cfo=None, feat_len=config.feat_len):
    cfos = []
    sts = []
    raw_signal_seg = []
    signal_start_time = []
    phase_amplitude = []
    labels = []
    local_labels = []
    ramp_segs = []
    time_stamp = []
    label_counter = 0
    origin_num = num
    pdr_info = []

    rt_num = 0
    rec_time = 0
    rt_time = 0
        
    for i, id in enumerate(ids):
        num = origin_num
        data_buf = []
        time_stamp_buf = []
        for sid in sample_ids:
            if origin_id is None:
                f = np.load(prefix.format(sid, id))
            else:
                f = np.load(prefix.format(sid, origin_id[id]))
            data_buf.append(f["arr_0"])
            time_stamp_buf.append(f["arr_1"])
            if "arr_2" in f:
                pdr_info.append(f["arr_2"])
        raw_data = np.vstack(data_buf)
        time_stamp_temp = np.concatenate(time_stamp_buf)
        rt_num += raw_data.shape[0]
        st = time.time()
        filter_signal = utils.filter(raw_data)
        if pre_cfo is not None:
            comp_raw_data = utils.cfo_compensate(raw_data, pre_cfo[i:i+1]) 
            comp_signal = utils.filter(comp_raw_data)
        else:
            comp_signal = filter_signal
        re_et = time.time()
        comp_phase_diff = utils.get_phase_diff(comp_signal)
        preamble_idx, _ = utils.find_preamble(comp_phase_diff, native=native, compare_num=4)
        phase = utils.get_phase(filter_signal)
        cfo, _, _ = utils.get_cfo(phase, preamble_idx)
        # filter_signal, _ = utils.normalize(filter_signal)
        ents = fe.get_ramp_seg_cfo(filter_signal, preamble_idx, cfo, raw_data, native=native, derivate_thres=5e-3, hight_diff_thres=0.2, feat_len=feat_len, time_stamp=None, preamble_bias=bias)
        et = time.time()
        rt_time += et - re_et
        rec_time += re_et - st
        # ents = fe.get_ramp_seg_old(filter_signal, preamble_idx, cfo, native)
        st = ents[2]
        if num == -1:
            num = ents[0].shape[0]
        cfos.append(cfo[ents[4]][:num])
        time_stamp.append(time_stamp_temp[ents[4]][:num])
        sts.append(st[:num])
        labels.append(np.array([label_counter for _ in range(num)]))
        if origin_id is not None:
            local_labels.append(np.array([id for _ in range(num)]))
        else:
            local_labels.append(np.array([i for _ in range(num)]))
        ramp_segs.append(ents[0][:num])
        raw_signal_seg.append(ents[1][:num])
        signal_start_time.append(ents[2][:num])
        label_counter += 1
        print(i, ents[0].shape)
    # print("->", feat_len, rt_time / rt_num)
    cfos = np.concatenate(cfos)
    sts = np.concatenate(sts)
    labels = np.concatenate(labels)
    local_labels = np.concatenate(local_labels)
    ramp_segs = np.vstack(ramp_segs)
    raw_signal_seg = np.vstack(raw_signal_seg)
    signal_start_time = np.concatenate(signal_start_time)
    if origin_id is not None:
        one_hot_label = np.zeros([cfos.shape[0], len(origin_id)], dtype=np.float32)
    else:
        one_hot_label = np.zeros([cfos.shape[0], len(ids)], dtype=np.float32)
    one_hot_label[(np.arange(cfos.shape[0]), local_labels)] = 1
    time_stamp = np.concatenate(time_stamp)
    if len(pdr_info) > 0:
        pdr_info = np.array(pdr_info)
        pdr = np.sum(pdr_info[:, 1]) / np.sum(pdr_info[:, 2])
    else:
        pdr = -1
    return cfos / 1000, sts / config.sample_rate, labels, local_labels, ramp_segs, one_hot_label, raw_signal_seg, signal_start_time, time_stamp, rec_time / rt_num, rt_time / rt_num, pdr
"""
0 cfos / 1000, 
1 sts / config.sample_rate, 
2 labels, 
3 local_labels, 
4 ramp_segs, 
5 one_hot_label, 
6 raw_signal_seg, 
7 signal_start_time, 
8 time_stamp
"""



# if __name__ == "__main__":
    # sample_num = -1
    # features = []
    # paths = {"nrf":"/data/blueprint/raw_data/distance/d=1m_ttt_{}_nodeid_{}_chan=37.npz", 
    #         "esp":"/data/blueprint/raw_data/0630_data/esp_raw/d=1m_ttt_{}_espid_{}_chan=37.npz",}
    # node_id = {"nrf":[i for i in range(1, 9)], 
    #         "esp":[i for i in range(9)],}
    # bias =  {"nrf":0,
    #         "esp":0,}

    # for key, value in paths.items():
    #     if key == "nrf":
    #         f = get_features(value, [i for i in range(2, 6)], node_id[key], sample_num, False, bias[key],  cfo_list)
    #     else:
    #         f = get_features(value, [i for i in range(2, 6)], node_id[key], sample_num, True, bias[key])
    #     np.savez("./data/" + key + "_basics.npz", cfo=f[0], st=f[1], label=f[2], local_label=f[3])
    #     cluster_info, _ = cluster_cfo(f[0])
    #     st_info = np.array([np.mean(f[1]), np.std(f[1])])
    #     np.savez("./data/" + key + "_cluster_info.npz", st_info=st_info, cluster_info=cluster_info)

