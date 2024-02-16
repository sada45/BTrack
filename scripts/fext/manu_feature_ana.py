import numpy as np
import utils
import feature_extractor as fe
import matplotlib.pyplot as plt
import scipy

nrf_cfo = np.array([108090.01250352601, 113508.6613375386, 110338.2471902627, -12720.419896746136, 113164.02264103664, -14845.531992728753, 8933.615425365566, -18704.24309504716, -10890.547783049955, 112777.59449315938, 106871.18719910384, -7755.785061007704])

# def mahalanobis_distance(x, clusters):
#     d = np.abs(x[:, np.newaxis] - clusters[:, 0])
#     md = d / clusters[:, 1]
#     return md


# st_clusters = []
# data = np.load("./data/nrf_cluster_info.npz")
# st_clusters.append(data["st_info"])

# def st_filter(sts, thres=3):
#     distance = mahalanobis_distance(sts, st_clusters)
#     min_idx = np.argmin(distance, axis=1)
#     min_val = np.min(distance, axis=1)
#     valid_idx = np.where(min_val<=thres)[0]  # Remove the alien devices 
#     min_idx = min_idx[valid_idx]
#     # min_val = min_val[valid_idx]
#     return distance[valid_idx, :], valid_idx


def tt(f, native):
    data = np.load(f)
    raw_data = data["arr_0"][:4350]
    time_stamp = data["arr_1"][:4350]
    filter_data = utils.filter(raw_data)
    # comp_signal = filter_data
    comp_signal = utils.cfo_compensate(filter_data, nrf_cfo[0])
    comp_phase_diff = utils.get_phase_diff(comp_signal)
    preamble_idx, _ = utils.find_preamble(comp_phase_diff, native=native, compare_num=32)
    phase = utils.get_phase(filter_data)
    cfo, _, _ = utils.get_cfo(phase, preamble_idx)
    ents = fe.get_ramp_seg_cfo(filter_data, preamble_idx, cfo, raw_data, native=native, derivate_thres=5e-3, hight_diff_thres=0.2, time_stamp=time_stamp, preamble_bias=0)
    ramp_fe = ents[0]
    
    return ramp_fe, ents[2], cfo[ents[4]], time_stamp[ents[4]]

feature_vectors = None
num = [0]

# ramp_fe, signal_start_time, cfo_0, time_stamp = tt("/data/blueprint/raw_data/0711_data/others/d=0m_ttt_4_nodeid_1_chan=37.npz", False)
# ramp_fe, signal_start_time, cfo_0, time_stamp = tt("/data/blueprint/raw_data/0711_data/others/d=1m_ttt_4_nodeid_1_chan=37.npz", False)

# temp_f = np.load("/data/blueprint/raw_data/0711_data/others/nrf52840_tmp_nodeid_1_0m.npz")
# temp_f = np.load("/data/blueprint/raw_data/0711_data/others/nrf52840_tmp_nodeid_1.npz")

ramp_fe, signal_start_time, cfo_0, time_stamp = tt("/data/blueprint/raw_data/0711_data/others/d=0m_ttt_4_nodeid_1_chan=37.npz", False)
temp_f = np.load("/data/blueprint/raw_data/0711_data/others/nrf52840_tmp_nodeid_1_0m.npz")

temperature = temp_f["arr_0"]
temperature_t = temp_f["arr_1"]

ramp_fe, time_stamp, cfo, features = fe.manual_feature_ext(ramp_fe[:, 4, :], time_stamp, cfo_0)


start_time = max(np.min(temperature_t), np.min(time_stamp))
end_time = min(np.max(temperature_t), np.max(time_stamp))
time_stamp_idx = np.where((time_stamp<=end_time) & (time_stamp>=start_time))[0]
temperature_t_idx = np.where((temperature_t<=end_time) & (temperature_t>=start_time))[0]
time_stamp = time_stamp[time_stamp_idx]
temperature_t = temperature_t[temperature_t_idx]
temperature = temperature[temperature_t_idx]
time_stamp -= start_time  # Start from 0
temperature_t -= start_time

# t_diff = time_stamp[1:] - time_stamp[0:-1]
# temp_t_start_idx = []
# temp_t_end_idx = []
# temp_buf = []
# gap_idx = np.where(t_diff>=1)[0]
# for i in gap_idx:
#     cur_time = time_stamp[i]
#     next_time = time_stamp[i+1]
#     temp_t_start_idx.append(np.where(temperature_t)>=cur_time)[0][0]
#     temp_t_end_idx.append(np.where(temperature_t)<=next_time)[0][0]
#     time_stamp[i+1:] = time_stamp[i+1:] - t_diff[i]

# new_tt = []





# _, st_valid_idx = st_filter(sts, )
# main_trough_width, overshots, min_idx, min_val, stable_val, stable_idx, diff_min_idx, diff_max_idx, diff_min_value, diff_max_value
mf = [features[0], features[2], features[3], features[4], cfo]
titles = ["Trough width", "min_idx", "min_val", "stable val", "cfo"]
ylims = [[25, 75], [360, 390], [-1.5, 0], [], [85000, 115000]]
plt.figure(figsize=(30, 30))
x = np.arange(ramp_fe.shape[0])
for i in range(5):
    fe = mf[i][time_stamp_idx]
    std = np.sqrt(np.var(fe))
    mean = np.mean(fe)
    up = mean + 3 * std
    down = mean - 3 * std
    norm_idx = np.where(time_stamp<100)[0]
    hightemp_idx = np.where(time_stamp>300)[0]
    print("=================")
    print(titles[i], "norm", np.mean(fe[norm_idx]), np.std(fe[norm_idx]))
    print("hightemp", np.mean(fe[hightemp_idx]), np.std(fe[hightemp_idx]))
    valid_idx = np.where((fe<=up)&(fe>=down))[0]
    t = time_stamp[valid_idx]
    f = fe[valid_idx]
    serial_idx = np.argsort(t)
    save_data = np.vstack([t[serial_idx], f[serial_idx]])
    np.savetxt("./output/1_temp_feature_{}.csv".format(titles[i]), save_data)
    plt.subplot(6, 1, i+1)
    plt.plot(t[serial_idx], f[serial_idx])
    # plt.plot(f[serial_idx])
    if len(ylims[i]) > 0:
        plt.ylim(ylims[i])
    plt.title(titles[i])

plt.subplot(6, 1, 6)
# temperature = scipy.signal.savgol_filter(temperature, window_length=61, polyorder=3)
plt.plot(temperature_t, temperature)
np.savetxt("./output/1_temp_features_temperature.csv", np.vstack([temperature_t, temperature]))
plt.savefig("./test_figs/features.pdf")
plt.close()



# ramp_fe, cfo, time_stamp = tt("/data/blueprint/raw_data/temperature/nrf52840_raw/nodeid=6.npz", False)


# diff = ramp_fe[:, 1:] - ramp_fe[:, 0:-1]
# for i in range(20):
#     plt.figure()
#     plt.plot(ramp_fe[i, :])
#     plt.savefig("./fe/esp_0_" + str(i) + ".pdf")
#     plt.close()

# for nodeid in range(1, 7):
#     ramp_fe, cfo, time_stamp = tt("./raw_data/nrf52840_raw/d=0_1_nodeid_" + str(nodeid) + "_chan=37.npz", False)
#     ramp_fe, time_stamp, cfo, features = fe.manual_feature_ext(ramp_fe, time_stamp, cfo)
#     features = np.array(features).T
#     print(features)
#     if feature_vectors is None:
#         feature_vectors = features
#     else:
#         feature_vectors = np.vstack([feature_vectors, features])
#     num.append(ramp_fe.shape[0])


# cum_num = np.cumsum(num)
# pca = PCA(n_components=2)
# pca.fit(feature_vectors)
# print(pca.explained_variance_ratio_)
# x = pca.transform(feature_vectors)
# plt.figure()
# for i in range(len(num)-1):
#     plt.scatter(x[cum_num[i]:cum_num[i]+32, 0], x[cum_num[i]:cum_num[i]+32, 1], marker=".", label=str(i+1))
# plt.legend()
# plt.savefig("./fe/pca.pdf")
# plt.close()

# np.savetxt("./output/manul_feature_pca.csv", feature_vectors, delimiter=",")






