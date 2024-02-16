import numpy as np
import matplotlib.pyplot as plt
import scripts.fext.utils as utils
import config
from feature_extractor import *

# data = np.load("./raw_data/esp_raw/d=0_1_espid_0_chan=38.npz")["arr_0"]
# data = np.load("./raw_data/nrf52840_raw/d=0_nodeid_2_chan=38.npz")["arr_0"]
data = np.load("./raw_data/nrf52840_raw/d=0_ramp_time=80_nodeid_2_chan=37.npz")["arr_0"]
filtered_data = utils.filter(data)

phase_diff = utils.get_phase_diff(filtered_data)
phase = utils.get_phase(filtered_data)
phase_cum_amp = utils.get_phase_cum(filtered_data)
preamble_idx, corrlation = utils.find_preamble(phase_diff, False)

# print(preamble_idx[5])
# b = utils.decode(filtered_data[5, ], preamble_idx[5])
i = 0
# while i < len(b):
#     print(b[i:i+8])
#     i += 8
# for i in range(64):
#     plt.figure(figsize=(12, 12))
#     plt.subplot(2, 1, 1)
#     plt.plot(phase_cum_amp[i, :2000])
#     x = np.array([preamble_idx[i]+config.sample_pre_symbol*j for j in range(9)], dtype=np.int32)
#     plt.plot(x, phase_cum_amp[i, x], linestyle=None, marker='o')
#     plt.subplot(2, 1, 2)
#     plt.plot(filtered_data[i, :2000].real)
#     plt.plot(filtered_data[i, :2000].imag)
#     plt.savefig("./figs/esp_preamble_" + str(i) + ".pdf")
#     plt.close()


cfo, slope, slope_amp = get_cfo(phase, preamble_idx, phase_cum_amp)
start_point = get_ramp_feature(filtered_data, preamble_idx, phase_cum_amp, slope_amp)
print(start_point)
avg_real = utils.normalize(utils.avg_filter(filtered_data.real))
avg_imag = utils.normalize(utils.avg_filter(filtered_data.imag))
# for i in range(64):
#     plt.figure(figsize=(12, 18))
#     plt.subplot(3, 1, 1)
#     plt.plot(phase_cum_amp[i, :700])
#     plt.plot([start_point[i]], [phase_cum_amp[i, start_point[i]]], marker="o", markersize=2)
#     plt.subplot(3, 1, 2)
#     plt.plot(avg_real[i, :700])
#     plt.plot(avg_imag[i, :700])
#     plt.plot([start_point[i], start_point[i]], [np.min(avg_real[i, :700]), np.max(avg_real[i, :700])])
#     plt.subplot(3, 1, 3)
#     plt.plot(filtered_data[i, :700].real)
#     plt.plot(filtered_data[i, :700].imag)
#     plt.plot([start_point[i], start_point[i]], [np.min(filtered_data[i, :700].real), np.max(filtered_data[i, :700].real)])
#     plt.savefig("./figs/esp_start_point_" + str(i) + ".pdf")
#     plt.close()
def get_ramp_feature(signal, preamble_idx, phase_cum, phase_slope, derivate_thres=1e-3, hight_diff_thres=0.1, feat_len=500):
    # First, we find where the rammping up started
    avg_filtered_real = utils.normalize(utils.avg_filter(signal.real, w=5))
    avg_filtered_imag = utils.normalize(utils.avg_filter(signal.imag, w=5))
    avg_filtered_phase_cmu = utils.avg_filter(phase_cum, w=5)
    diff_real = avg_filtered_real[:, 1:] - avg_filtered_real[:, :-1]
    diff_imag = avg_filtered_imag[:, 1:] - avg_filtered_imag[:, :-1]
    signal_start_point = np.zeros(signal.shape[0], dtype=np.int32)
    for i in range(signal.shape[0]):
        # find all index where the derivation is 0. These points are peaks or bottoms
        real_flat_point_idx = zero_derivation_idx(diff_real[i, :], thres=derivate_thres)
        real_flat_point_val = avg_filtered_real[i, real_flat_point_idx]
        # then, we calculate the difference of consecutive flat points.
        # the beginning of the raw signal is relatively flat and the difference is small
        # when the signal starts transmitting, the amplitude suddently changed
        real_start_idx, _ = find_signal_start(real_flat_point_val, thres=hight_diff_thres)
        real_start = real_flat_point_idx[real_start_idx]
        imag_flat_point_idx = zero_derivation_idx(diff_imag[i, :])
        imag_flat_point_val = avg_filtered_imag[i, imag_flat_point_idx]
        imag_start_idx, _ = find_signal_start(imag_flat_point_val, thres=hight_diff_thres)
        imag_start = imag_flat_point_idx[imag_start_idx]
        if real_start == -1 or imag_start == -1:
            signal_start_point[i] = 0
        else:
            signal_start_point[i] = np.min([real_start, imag_start])
    # return signal_start_point

    # Then, we remove the effect of the CFO, normalize the signal around the rammping up
    start_point = preamble_idx - feat_len
    seg_len = feat_len + 8 * config.sample_pre_symbol
    # start_point[np.where(start_point<0)] = 0
    signal_start_point_in_seg = signal_start_point - start_point
    upper_preamble_idx = np.array([0, 2, 4, 6, 8], dtype=np.int32) * config.sample_pre_symbol
    lower_preamble_idx = np.array([1, 3, 5, 7], dtype=np.int32) * config.sample_pre_symbol
    rampping_up_signals = np.zeros([signal.shape[0], feat_len], dtype=np.float64)
    for i in range(signal.shape[0]):
        # a = (start_point[i], start_point[i]+feat_len+8*config.sample_pre_symbol+1)
        phase_cum_seg = avg_filtered_phase_cmu[i, start_point[i]:start_point[i]+feat_len+8*config.sample_pre_symbol+1]
        cfo_compensation = -phase_slope[i] * np.arange(0, len(phase_cum_seg)-signal_start_point_in_seg[i])
        phase_cum_seg[signal_start_point_in_seg[i]: ] = phase_cum_seg[signal_start_point_in_seg[i]: ] + cfo_compensation
        phase_amplitude = np.mean(phase_cum_seg[feat_len+upper_preamble_idx]) - np.mean(phase_cum_seg[feat_len+lower_preamble_idx])
        rampping_up_signals[i, :] = phase_cum_seg[:feat_len] / phase_amplitude
        if i < 10:
            plt.figure(figsize=(12, 18))
            plt.subplot(3, 1, 1)
            plt.plot(phase_cum[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol])
            plt.plot(np.array([signal_start_point_in_seg[i], signal_start_point_in_seg[i]]), np.array([np.min(phase_cum[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].real), np.max(phase_cum[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].real)]))
            plt.xlim([0, seg_len])
            plt.subplot(3, 1, 3)
            plt.plot(rampping_up_signals[i, :])
            # plt.xlim([0, seg_len])
            plt.subplot(3, 1, 2)
            plt.plot(signal[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].real)
            plt.plot(signal[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].imag)
            plt.plot(np.array([signal_start_point_in_seg[i], signal_start_point_in_seg[i]]), np.array([np.min(signal[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].real), np.max(signal[i, start_point[i]:preamble_idx[i]+8*config.sample_pre_symbol].real)]))
            plt.xlim([0, seg_len])
            plt.savefig("./figs/ramp_ramp=80_2_" + str(i) + ".pdf")
            plt.close()
    return signal_start_point




#################################
# PCA 
# node_cfo_each_chan = np.zeros([8, 40, 64], dtype=np.float64)
# for node_id in range(1, 9):
#     cfo_list = []
#     for chan in range(40):
#         data = np.load("./raw_data/d=50_nodeid_" + str(node_id) + "_chan=" + str(chan) + ".npz")["arr_0"]
#         cfo_list.append(get_features(data))
#         node_cfo_each_chan[node_id-1, chan, :] = get_features(data)


# node_cfo_each_chan = np.ascontiguousarray(node_cfo_each_chan.transpose(0, 2, 1))
# for i in range(8):
#     np.savetxt("node=" + str(i+1) + ".csv", node_cfo_each_chan[i, :, :], delimiter=",")


# np.save("./cfo_test.npy", node_cfo_each_chan)

# node_cfo_each_chan = np.load("./cfo.npy")
# node_cfo_test = np.load("./cfo_test.npy")
# train_cfo = node_cfo_each_chan[:, :, :].reshape([64*8, 40])
# test_cfo = node_cfo_test[:, :, :].reshape([64*8, 40])
# cfos = np.vstack([train_cfo, test_cfo])
# pca = PCA(n_components=2)
# pca.fit(cfos)
# print(pca.explained_variance_ratio_)
# ld_cfo = pca.transform(cfos)
# plt.figure(figsize=(12, 12))
# for i in range(8):
#     plt.scatter(ld_cfo[i*64:i*64+16, 0], ld_cfo[i*64:i*64+16, 1], marker='o', label="id="+str(i+1))
# plt.legend()
# plt.savefig("./figs/pca.pdf")

# plt.figure(figsize=(12, 12))
# for i in range(8):
#     plt.scatter(ld_cfo[i*64:i*64+16, 0], ld_cfo[i*64:i*64+16, 1], marker='o', label="id="+str(i+1))
# plt.legend()
# plt.savefig("./figs/pca_close.pdf")

# train_labels = np.zeros(64 * 8)
# for i in range(8):
#     train_labels[i*64: (i+1)*64] = i + 1

# print("start trainng")
# clf = svm.SVC(C=1,kernel='poly',gamma=0.01)
# clf.fit(train_cfo, train_labels)
# predict_label = clf.predict(test_cfo)
# for i in range(8):
#     correct_num = len(np.where(predict_label[i*64: (i+1)*64]==train_labels[i*64: (i+1)*64])[0])
#     print(i+1, correct_num / 64)
# correct_num = len(np.where(predict_label == train_labels)[0])
# print(correct_num / test_cfo.shape[0])

