import numpy as np
import matplotlib.pyplot as plt
import utils as utils
# from sklearn import svm
# from sklearn.decomposition import PCA
import config
import os
import scipy

error_num = 0

phase = None

# error_file = os.listdir("./errors/")
# for f in error_file:
#     os.remove("./errors/" + f)

# Used to find the start point of the I/Q raw signal
def zero_derivation_idx(diff, thres=1e-3, interval_thres=5, bias=0, last=True):
    zeros_idx = np.where((diff>=-thres) & (diff<=thres))[0]
    if len(zeros_idx) > 0:
        peaks = []
        buf = [zeros_idx[0]]
        for j in range(1, len(zeros_idx)):
            if len(buf) == 0 or zeros_idx[j] - buf[-1] <= interval_thres:
                buf.append(zeros_idx[j])
            else:
                if last:
                    peaks.append(buf[-1] + bias)
                else:
                    peaks.append(buf[len(buf) // 2] + bias)
                buf = []
        if len(buf) > 0:
            if last:
                peaks.append(buf[-1] + bias)
            else:
                peaks.append(buf[len(buf) // 2] + bias)
            buf = []
        return np.array(peaks, dtype=np.int32)
    else:
        return None

def find_peak_and_troughss(signal):
    peaks, _ = scipy.signal.find_peaks(signal)
    troughs, _ = scipy.signal.find_peaks(-signal)
    return peaks, troughs

def find_signal_start(signal, preamble_idx, feat_len, hight_diff_thres=0.2, derivation_thres=1e-3):
    # flat_point_idx, flat_point_val, 
    signal_diff = signal[preamble_idx-feat_len:preamble_idx] - signal[preamble_idx-feat_len-1: preamble_idx-1] 
    flat_point_idx = zero_derivation_idx(signal_diff, thres=derivation_thres, interval_thres=3)
    if flat_point_idx is None:
        return -1, -1
    flat_point_val = signal[flat_point_idx + preamble_idx - feat_len]
    if len(flat_point_val) < 2:
        return -1, -1
    # plt.figure(figsize=(12, 6))
    # plt.plot(signal[preamble_idx-config.feat_len:preamble_idx])
    # for i in range(len(flat_point_idx)):
    #     plt.plot([flat_point_idx[i], flat_point_idx[i]], [-0.5, 0.5])
    # plt.savefig("./test_figs/t_" + str(c) + ".pdf")
    # c += 1
    # plt.close()
    diff = np.abs(flat_point_val[1:] - flat_point_val[:-1])
    sudden_change_idxs = np.where(diff>=hight_diff_thres)[0]
    if len(sudden_change_idxs) == 0:
        return -1, -1
    sudden_change_idx = sudden_change_idxs[0]
    return flat_point_idx[sudden_change_idx] + preamble_idx - feat_len, diff[sudden_change_idx]

counter = 0

# def get_ramp_seg_cfo(signal, preamble_idx, cfo, raw_signal, native=False, derivate_thres=1e-3, hight_diff_thres=0.2, feat_len=config.feat_len, time_stamp=None, preamble_bias=0):
#     # preamble bias: some of the devices can transmit 2bytes preamble (e.g., raspberry 3b+)
#     global error_num, counter
#     compensated_signal = utils.cfo_compensate(signal, cfo)
#     raw_compensated_signal = utils.cfo_compensate(raw_signal, cfo)
#     phase_diff = utils.get_phase_diff(compensated_signal)
#     # First, we find where the rammping up started
#     avg_filtered_real, scale_real = utils.normalize(scipy.signal.savgol_filter(compensated_signal.real, window_length=81, polyorder=5))
#     avg_filtered_imag, scale_imag = utils.normalize(scipy.signal.savgol_filter(compensated_signal.imag, window_length=81, polyorder=5))

#     signal_start_point = np.zeros(signal.shape[0], dtype=np.int32)
#     upper_preamble_idx = np.array([2, 4, 6, 8], dtype=np.int32) * config.sample_pre_symbol
#     lower_preamble_idx = np.array([1, 3, 5, 7], dtype=np.int32) * config.sample_pre_symbol

#     # First, we find the signal start point according to the I/Q raw signal 
#     for i in range(signal.shape[0]):
#         if preamble_idx[i] == 0 or preamble_idx[i] <= 0:
#             continue
#         real_start, _ = find_signal_start(avg_filtered_real[i, :], preamble_idx[i], feat_len, hight_diff_thres=hight_diff_thres, derivation_thres=derivate_thres)
#         imag_start, _ = find_signal_start(avg_filtered_imag[i, :], preamble_idx[i], feat_len, hight_diff_thres=hight_diff_thres, derivation_thres=derivate_thres)
#         if real_start == -1 and imag_start == -1:
#             signal_start_point[i] = 0
#         else:
#             if real_start == -1:
#                 signal_start_point[i] = imag_start
#             elif imag_start == -1:
#                 signal_start_point[i] = real_start
#             else:
#                 signal_start_point[i] = np.min([real_start, imag_start])
#             # print(real_start, imag_start, signal_start_point[i])

#     start_point = preamble_idx - feat_len - preamble_bias
#     rampup_len = preamble_idx - signal_start_point
#     signal_start_point_in_seg = signal_start_point - start_point
   
#     valid_idx = []  # Some signal may be incorrect (e.g., received a highly distored signal)
#     for i in range(signal.shape[0]):
#         if preamble_idx[i] == 0 or signal_start_point_in_seg[i] < 0 or signal_start_point[i] > preamble_idx[i] or rampup_len[i] <= preamble_bias:
#             # if time_stamp is not None:
#             #     print("{0}th signal, with start point in seg {1}, signal_start {2}, preamble_idx{3}, error{4}, ramplen{5}, time_stamp{6}".format(i, signal_start_point_in_seg[i], signal_start_point[i], preamble_idx[i], error_num, rampup_len[i], time_stamp[i]))
#             # else:
#             #     print("{0}th signal, with start point in seg {1}, signal_start {2}, preamble_idx{3}, error{4}, ramplen{5}".format(i, signal_start_point_in_seg[i], signal_start_point[i], preamble_idx[i], error_num, rampup_len[i]))

#             # plt.figure(figsize=(12, 12))
#             # plt.subplot(2, 1, 1)
#             # plt.plot(avg_filtered_real[i, :preamble_idx[i]+64*config.sample_pre_symbol])
#             # plt.plot(avg_filtered_imag[i, :preamble_idx[i]+64*config.sample_pre_symbol])
#             # plt.plot([signal_start_point[i], signal_start_point[i]], [-0.5, 0.5])
#             # plt.ylim([-0.5, 0.5])
#             # plt.subplot(2, 1, 2)
#             # p = utils.get_phase_cum_1d(signal[i, preamble_idx[i]-feat_len:preamble_idx[i]+160])
#             # plt.plot(p)
#             # plt.savefig("./errors/error_" + str(error_num) + ".pdf")
#             # error_num += 1
#             # plt.close()
#             continue
#         # plt.figure(figsize=(12, 12))
#         # plt.subplot(2, 1, 1)
#         # plt.plot(avg_filtered_real[i, :preamble_idx[i]+64*config.sample_pre_symbol])
#         # plt.plot(avg_filtered_imag[i, :preamble_idx[i]+64*config.sample_pre_symbol])
#         # plt.plot([signal_start_point[i], signal_start_point[i]], [-0.5, 0.5])
#         # plt.ylim([-0.5, 0.5])
#         # plt.subplot(2, 1, 2)
#         # p = utils.get_phase_cum_1d(signal[i, preamble_idx[i]-feat_len:preamble_idx[i]+160])
#         # plt.plot(p)
#         # plt.savefig("./errors/error_" + str(error_num) + ".pdf")
#         # error_num += 1
#         # plt.close()
#         # compensated_signal[i, signal_start_point[i]:] = utils.cfo_compensate_1d(compensated_signal[i, signal_start_point[i]:], cfo[i])  # CFO compensation 
#         new_preamble_idx, _ = utils.find_preamble(phase_diff[i:i+1, :], native, preamble_idx[i:i+1])
#         preamble_idx[i] = new_preamble_idx[0]
#         if new_preamble_idx[0] != 0:
#             valid_idx.append(i)
#     preamble_idx = preamble_idx[valid_idx]
#     # The previously finded preamble index may have errors, we need to update it 
#     # phase_diff = utils.get_phase_diff(compensated_signal[valid_idx, :])
#     # preamble_idx, _ = utils.find_preamble(phase_diff, native, preamble_idx[valid_idx])
#     start_point = preamble_idx - feat_len - preamble_bias
#     signal_start_point_in_seg = signal_start_point[valid_idx] - start_point
    
#     rampping_up_signals = np.zeros([signal.shape[0], len(utils.ndlpf_a), feat_len], dtype=np.float64)  # The extracted rampping up feature
#     raw_signal_seg = np.zeros([signal.shape[0], feat_len], dtype=np.complex64)
#     phase_amplitude = np.zeros(signal.shape[0], dtype=np.float32)

#     for i in range(len(valid_idx)):
#         # compensated_signal[valid_idx[i], :preamble_idx[i]+1] 
#         nd_signal = utils.near_dc_lpf(raw_compensated_signal[valid_idx[i], preamble_idx[i]-feat_len-preamble_bias:preamble_idx[i]-preamble_bias+1])  
#         # Since we have removed the CFO, we can use a LPF with 200KHz bandwidth
#         rampping_up_signals[i, :, :] = utils.get_phase_cum(nd_signal)
#         phase_cum_seg = utils.get_phase_cum_1d(compensated_signal[valid_idx[i], start_point[i]:start_point[i]+feat_len+8*config.sample_pre_symbol+2])
#         phase_amplitude[i] = np.mean(phase_cum_seg[feat_len+upper_preamble_idx]) - np.mean(phase_cum_seg[feat_len+lower_preamble_idx])
#         rampping_up_signals[i, :, :] = rampping_up_signals[i, :, :] / phase_amplitude[i]
#         # rampping_up_signals[i, :] = rampping_up_signals[i, :] - rampping_up_signals[i, 0]
#         # rampping_up_signals[i, signal_start_point_in_seg[i]:] -=  rampping_up_signals[i, signal_start_point_in_seg[i]]
#         # rampping_up_signals[i, :signal_start_point_in_seg[i]] = 0
#         for j in range(len(utils.ndlpf_a)):
#             rampping_up_signals[i, j, :] -= rampping_up_signals[i, j, signal_start_point_in_seg[i]]
#         raw_signal_seg[i, :] = compensated_signal[valid_idx[i], start_point[i]:start_point[i]+feat_len]

#     valid_idx = np.array(valid_idx, dtype=np.int32)
#     return rampping_up_signals[:len(valid_idx), :], raw_signal_seg[:len(valid_idx), :], signal_start_point_in_seg, signal_start_point[valid_idx], valid_idx, preamble_idx, (scale_real[valid_idx] + scale_imag[valid_idx]) / 2, phase_amplitude[:len(valid_idx)]
"""
0   rampping_up_signals[:len(valid_idx), :], 
1   raw_signal_seg[:len(valid_idx), :], 
2   signal_start_point_in_seg, 
3   signal_start_point[valid_idx], 
4   valid_idx, 
5   preamble_idx, 
6   (scale_real[valid_idx] + scale_imag[valid_idx]) / 2, 
7   phase_amplitude[:len(valid_idx)]

"""

def get_ramp_seg_cfo(signal, preamble_idx, cfo, raw_signal, native=False, derivate_thres=1e-3, hight_diff_thres=0.2, feat_len=config.feat_len, time_stamp=None, preamble_bias=0):
    # preamble bias: some of the devices can transmit 2bytes preamble (e.g., raspberry 3b+)
    global error_num, counter
    compensated_signal = utils.cfo_compensate(signal, cfo)
    raw_compensated_signal = utils.cfo_compensate(raw_signal, cfo)
    phase_diff = utils.get_phase_diff(compensated_signal)
    # First, we find where the rammping up started
    avg_filtered_real, scale_real = utils.normalize(scipy.signal.savgol_filter(compensated_signal.real, window_length=81, polyorder=5))
    avg_filtered_imag, scale_imag = utils.normalize(scipy.signal.savgol_filter(compensated_signal.imag, window_length=81, polyorder=5))

    signal_start_point = np.zeros(signal.shape[0], dtype=np.int32)
    upper_preamble_idx = np.array([2, 4, 6, 8], dtype=np.int32) * config.sample_pre_symbol
    lower_preamble_idx = np.array([1, 3, 5, 7], dtype=np.int32) * config.sample_pre_symbol

    # First, we find the signal start point according to the I/Q raw signal 
    for i in range(signal.shape[0]):
        if preamble_idx[i] == 0 or preamble_idx[i] <= 0:
            continue
        real_start, _ = find_signal_start(avg_filtered_real[i, :], preamble_idx[i], feat_len, hight_diff_thres=hight_diff_thres, derivation_thres=derivate_thres)
        imag_start, _ = find_signal_start(avg_filtered_imag[i, :], preamble_idx[i], feat_len, hight_diff_thres=hight_diff_thres, derivation_thres=derivate_thres)
        if real_start == -1 and imag_start == -1:
            signal_start_point[i] = 0
        else:
            if real_start == -1:
                signal_start_point[i] = imag_start
            elif imag_start == -1:
                signal_start_point[i] = real_start
            else:
                signal_start_point[i] = np.min([real_start, imag_start])
            # print(real_start, imag_start, signal_start_point[i])

    start_point = preamble_idx - feat_len
    rampup_len = preamble_idx - signal_start_point
    signal_start_point_in_seg = signal_start_point - start_point
   
    valid_idx = []  # Some signal may be incorrect (e.g., received a highly distored signal)
    for i in range(signal.shape[0]):
        if preamble_idx[i] == 0 or signal_start_point_in_seg[i] < 0 or signal_start_point[i] > preamble_idx[i] or rampup_len[i] <= preamble_bias or preamble_idx[i]<feat_len:
            if time_stamp is not None:
                print("{0}th signal, with start point in seg {1}, signal_start {2}, preamble_idx{3}, error{4}, ramplen{5}, time_stamp{6}".format(i, signal_start_point_in_seg[i], signal_start_point[i], preamble_idx[i], error_num, rampup_len[i], time_stamp[i]))
            else:
                print("{0}th signal, with start point in seg {1}, signal_start {2}, preamble_idx{3}, error{4}, ramplen{5}".format(i, signal_start_point_in_seg[i], signal_start_point[i], preamble_idx[i], error_num, rampup_len[i]))

            # plt.figure(figsize=(12, 12))
            # plt.subplot(2, 1, 1)
            # plt.plot(avg_filtered_real[i, :preamble_idx[i]+64*config.sample_pre_symbol])
            # plt.plot(avg_filtered_imag[i, :preamble_idx[i]+64*config.sample_pre_symbol])
            # plt.plot([signal_start_point[i], signal_start_point[i]], [-0.5, 0.5])
            # plt.ylim([-0.5, 0.5])
            # plt.subplot(2, 1, 2)
            # p = utils.get_phase_cum_1d(signal[i, preamble_idx[i]-feat_len:preamble_idx[i]+160])
            # plt.plot(p)
            # plt.savefig("./errors/error_" + str(error_num) + ".pdf")
            # error_num += 1
            # plt.close()
            continue
        # compensated_signal[i, signal_start_point[i]:] = utils.cfo_compensate_1d(compensated_signal[i, signal_start_point[i]:], cfo[i])  # CFO compensation 
        new_preamble_idx, _ = utils.find_preamble(phase_diff[i:i+1, :], native, preamble_idx[i:i+1])
        preamble_idx[i] = new_preamble_idx[0]
        if new_preamble_idx[0] != 0:
            valid_idx.append(i)
    preamble_idx = preamble_idx[valid_idx]
    # The previously finded preamble index may have errors, we need to update it 
    # phase_diff = utils.get_phase_diff(compensated_signal[valid_idx, :])
    # preamble_idx, _ = utils.find_preamble(phase_diff, native, preamble_idx[valid_idx])
    start_point = preamble_idx - feat_len
    signal_start_point_in_seg = signal_start_point[valid_idx] - start_point
    
    rampping_up_signals = np.zeros([signal.shape[0], len(utils.ndlpf_a), feat_len], dtype=np.float64)  # The extracted rampping up feature
    raw_signal_seg = np.zeros([signal.shape[0], feat_len], dtype=np.complex64)
    phase_amplitude = np.zeros(signal.shape[0], dtype=np.float32)

    for i in range(len(valid_idx)):
        # compensated_signal[valid_idx[i], :preamble_idx[i]+1] 
        nd_signal = utils.near_dc_lpf(raw_compensated_signal[valid_idx[i], preamble_idx[i]-feat_len:preamble_idx[i]+1])  
        # Since we have removed the CFO, we can use a LPF with 200KHz bandwidth
        rampping_up_signals[i, :, :] = utils.get_phase_cum(nd_signal)
        phase_cum_seg = utils.get_phase_cum_1d(compensated_signal[valid_idx[i], start_point[i]:start_point[i]+feat_len+8*config.sample_pre_symbol+2])
        phase_amplitude[i] = np.mean(phase_cum_seg[feat_len+upper_preamble_idx]) - np.mean(phase_cum_seg[feat_len+lower_preamble_idx])
        rampping_up_signals[i, :, :] = rampping_up_signals[i, :, :] / phase_amplitude[i]
        # rampping_up_signals[i, :] = rampping_up_signals[i, :] - rampping_up_signals[i, 0]
        # rampping_up_signals[i, signal_start_point_in_seg[i]:] -=  rampping_up_signals[i, signal_start_point_in_seg[i]]
        # rampping_up_signals[i, :signal_start_point_in_seg[i]] = 0
        for j in range(len(utils.ndlpf_a)):
            # rampping_up_signals[i, j, :] -= rampping_up_signals[i, j, 0]
            rampping_up_signals[i, j, :] -= rampping_up_signals[i, j, signal_start_point_in_seg[i]]
            rampping_up_signals[i, j, :signal_start_point_in_seg[i]] = 0
        raw_signal_seg[i, :] = compensated_signal[valid_idx[i], start_point[i]:start_point[i]+feat_len]

    valid_idx = np.array(valid_idx, dtype=np.int32)
    return rampping_up_signals[:len(valid_idx), :], raw_signal_seg[:len(valid_idx), :], signal_start_point_in_seg, signal_start_point[valid_idx], valid_idx, preamble_idx, (scale_real[valid_idx] + scale_imag[valid_idx]) / 2, phase_amplitude[:len(valid_idx)]
"""
0   rampping_up_signals[:len(valid_idx), :], 
1   raw_signal_seg[:len(valid_idx), :], 
2   signal_start_point_in_seg, 
3   signal_start_point[valid_idx], 
4   valid_idx, 
5   preamble_idx, 
6   (scale_real[valid_idx] + scale_imag[valid_idx]) / 2, 
7   phase_amplitude[:len(valid_idx)]

"""


def esp_manual_feature_ext(rampup_seg, start_time, time_stamp, cfo, flat_thres=3e-3):
    valid_idx = []
    min_val1 = []
    min_val2 = []
    distance = []
    ratio = []

    for i in range(rampup_seg.shape[0]):
        seg = rampup_seg[i, start_time[i]:]
        diff = seg[1:] - seg[0:-1]
        local_max_idxs = np.where(diff>=0)[0]
        if len(local_max_idxs) > 0:
            local_max_idx = local_max_idxs[-1]
        else:
            continue
        seg = seg[:local_max_idx+1]
        peaks, _ = scipy.signal.find_peaks(-seg, distance=10)
        if len(peaks) == 2:
        #     valid_idx.append(i)
        #     plt.figure()
        #     s = rampup_seg[i, start_time[i]:]
        #     plt.plot(s)
        #     plt.plot(peaks, s[peaks], "x")
        #     plt.savefig("./fe/esp_fe_" + str(i) + ".pdf")
        #     plt.close()
            valid_idx.append(i)
            min_val1.append(seg[peaks[0]])
            min_val2.append(seg[peaks[1]])
            distance.append(peaks[1] - peaks[0])
            ratio.append(min_val1[-1] / min_val2[-1])
    return rampup_seg[valid_idx, :], time_stamp[valid_idx], cfo[valid_idx], [np.array(min_val1), np.array(min_val2), np.array(distance), np.array(ratio)]

def manual_feature_ext(rampup_seg, time_stamp, cfo, flat_thres=5e-3):
    min_idx = np.argmin(rampup_seg, axis=1)
    min_val = np.zeros(rampup_seg.shape[0])
    for i in range(rampup_seg.shape[0]):
        min_val[i] = rampup_seg[i, min_idx[i]]
    zero_idx = np.where(min_val==0)[0]
    print(len(zero_idx))
    
    rampup_seg = np.delete(rampup_seg, zero_idx, axis=0)
    min_idx = np.delete(min_idx, zero_idx)
    min_val = np.delete(min_val, zero_idx)
    time_stamp = np.delete(time_stamp, zero_idx, axis=0)
    cfo = np.delete(cfo, zero_idx, axis=0)

    
    diff = rampup_seg[:, 1:] - rampup_seg[:, 0:-1]
    diff_smooth = scipy.signal.savgol_filter(diff, window_length=81, polyorder=3, axis=1)
    diff_min_idx = np.argmin(diff_smooth, axis=1)
    diff_max_idx = np.argmax(diff_smooth[:, :500], axis=1)
    diff_min_value = diff_smooth[(np.arange(diff_smooth.shape[0]), diff_min_idx)]
    diff_max_value = diff_smooth[(np.arange(diff_smooth.shape[0]), diff_max_idx)]
    main_trough_width = diff_max_idx - diff_min_idx
    # diff2 = diff_smooth[:, 1:] - diff_smooth[:, 0:-1]
    stable_idx = np.zeros(rampup_seg.shape[0], dtype=np.int32)
    stable_val = np.zeros(rampup_seg.shape[0], dtype=np.float32)
    for i in range(rampup_seg.shape[0]):
        # plt.figure()
        # plt.plot(rampup_seg[i, :])
        # plt.plot(diff_smooth[i, :]*10)
        # plt.savefig("./fe/p_" + str(i) + ".pdf")
        # plt.close()
        flat_points = np.where(diff_smooth[i, diff_max_idx[i]:]<=flat_thres)[0][0] 
        flat_points = flat_points + diff_max_idx[i] + 1
        stable_idx[i] = flat_points
        stable_val[i] = np.mean(rampup_seg[i, flat_points:])
        # plt.figure(figsize=(12, 12))
        # plt.subplot(2, 1, 1)
        # plt.plot(rampup_seg[i, :])
        # plt.plot([stable_idx[i], stable_idx[i]], [np.min(rampup_seg[i, :]), np.max(rampup_seg[i, :])])
        # plt.subplot(2, 1, 2)
        # plt.plot(diff_smooth[i, :])
        # plt.plot([stable_idx[i], stable_idx[i]], [np.min(diff_smooth[i,:]), np.max(diff_smooth[i, :])])
        # plt.savefig("./fe/diff_" + str(i) + ".pdf")
        # plt.close()

    overshots = stable_val / min_val
    main_trough_width = main_trough_width / config.sample_pre_symbol
    min_idx = min_idx / config.sample_pre_symbol
    return rampup_seg, time_stamp, cfo, [main_trough_width, overshots, min_idx, min_val, stable_val, stable_idx, diff_min_idx, diff_max_idx, diff_min_value, diff_max_value]
        

    













# if __name__ == "__main__":
#     for nodeid in range(7, 8):
#         print("nodeid", nodeid)
#         data = np.load("./raw_data/0630_data/esp_raw/d=10m_ttt_3_espid_" + str(nodeid) + "_chan=37.npz")["arr_0"]
#         is_native = True
#         # data = np.load("./raw_data/0630_data/nrf52840_raw/d=1m_ttt_3_nodeid_" + str(nodeid) + "_chan=37.npz")["arr_0"]
#         # is_native = False
#         filtered_data = utils.filter(data)
#         phase_diff = utils.get_phase_diff(filtered_data)
#         phase = utils.get_phase(filtered_data)
#         phase_cum_amp = utils.get_phase_cum(filtered_data)
#         # for i in range(10):
#         #     plt.figure(figsize=(12, 12))
#         #     plt.subplot(2, 1, 1)
#         #     plt.plot(phase_cum_amp[i, :2000])
#         #     plt.subplot(2, 1, 2)
#         #     plt.plot(filtered_data[i, :].real)
#         #     plt.plot(filtered_data[i, :].imag)
#         #     plt.savefig("./test_figs/test_" + str(i) + ".pdf")
#         #     plt.close()
#         preamble_idx, _ = utils.find_preamble(phase_diff, is_native)
#         print(preamble_idx)
#         cfo, slope, slope_amp = utils.get_cfo(phase, preamble_idx, phase_cum_amp)
#         normalized_data, scale = utils.normalize(filtered_data)
#         ramp_seg, raw_signal_seg, start_point, signal_start_point, valid_idx, new_preamble_idx = get_ramp_seg_cfo(normalized_data, preamble_idx, cfo, native=is_native)
#         noise_seg = []
#         for i in range(len(valid_idx)):
#             noise_seg.append(data[valid_idx[i], :signal_start_point[i]])
#         snr = utils.cal_SNR(raw_signal_seg, noise_seg, scale[valid_idx])
#         print(snr)

#         # ramp_seg, start_point, start_point_origin = get_ramp_seg(filtered_data, preamble_idx,cfo)
#         # print(start_point)
#         for i in range(10):
#             plt.figure(figsize=(12, 18))
#             plt.subplot(3, 1, 1)
#             plt.plot(ramp_seg[i, :])
#             plt.subplot(3, 1, 2)
#             plt.plot(data[valid_idx[i], :1000].real)
#             plt.plot(data[valid_idx[i], :1000].imag)
#             # plt.plot([start_point_origin[i], start_point_origin[i]], [-1, 1])
#             plt.subplot(3, 1, 3)
#             plt.plot(phase_cum_amp[i, :])
#             plt.plot([preamble_idx[i], preamble_idx[i]], [np.min(phase_cum_amp[i, ]), np.max(phase_cum_amp[i, :])])
#             # print(i, start_point_origin[i], preamble_idx[i], new_preamble_idx[i])
#             plt.savefig("./figs/phase_52840=" + str(nodeid) + "_" + str(i) + ".pdf")
#             plt.close()


    # compensated_signal = utils.cfo_compensate(filtered_data, cfo)
    # compensated_phase_diff = utils.get_phase_diff(compensated_signal)
    # compensated_phase_cum = utils.get_phase_cum(compensated_signal)

    # for i in range(5):
    #     plt.figure(figsize=(12, 12))
    #     plt.subplot(2, 1, 1)
    #     plt.plot(phase_cum_amp[i, :])
    #     plt.subplot(2, 1, 2)
    #     plt.plot(compensated_phase_cum[i, :])
    #     plt.savefig("./test_figs/compen_" + str(i) + ".pdf")
    #     plt.close()