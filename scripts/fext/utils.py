import numpy as np
import scipy
import config
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scramble_table import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_aa_bit = np.array([0,0,0,1, 1,1,1,0, 0,1,1,0, 1,0,1,0, 0,0,1,0, 1,1,0,0, 0,1,0,0, 1,0,0,0], dtype=np.uint8)
native_target_aa_bit = np.array([0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)

b, a = scipy.signal.butter(8, 1.5e6 / config.sample_rate, "lowpass")
ndlpf_b = []
ndlpf_a = []
# , 1e6, 1.2e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6, 2.2e6, 2.4e6
for bd in [8e5, 1e6, 1.2e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6, 2.2e6, 2.4e6]:
    # b, a = scipy.signal.butter(8, 1.5e6 / config.sample_rate, "lowpass")
    nb, na = scipy.signal.butter(8, bd / config.sample_rate, "lowpass")
    ndlpf_b.append(nb)
    ndlpf_a.append(na)
# b, a = scipy.signal.butter(8, 2e6 / config.sample_rate, "lowpass")
# ndlpf_b = []
# ndlpf_a = []
# for bd in [8e5, 1e6, 1.2e6, 1.4e6, 1.6e6, 1.8e6, 2e6, 2.2e6, 2.4e6]:
#     b, a = scipy.signal.butter(8, 1.5e6 / config.sample_rate, "lowpass")
#     # b, a = scipy.signal.butter(8, bd / config.sample_rate, "lowpass")
#     ndlpf_b.append(b)
#     ndlpf_a.append(a)
adv_aa = 0x8e89bed6
# nrf_aa = 0x12345678
nrf_aa = 0x8e89bed8

def get_preamble_mask(native=False):
    mask = torch.ones(39 * config.sample_pre_symbol, dtype=torch.float32)
    for i in range(1, 7, 2):
        mask[i * config.sample_pre_symbol: (i+1) * config.sample_pre_symbol] = -1
    
    if native:
        aa = adv_aa
    else:
        aa = nrf_aa
        return mask[:7*config.sample_pre_symbol].view((1, 1, -1)).contiguous().to(device)
    
    for i in range(32):
        if (aa >> i) & 1:
            mask[(i+7)*config.sample_pre_symbol:(i+8)*config.sample_pre_symbol] = 1
        else:
            mask[(i+7)*config.sample_pre_symbol:(i+8)*config.sample_pre_symbol] = -1
    return mask.view((1, 1, -1)).contiguous().to(device)

def get_preamble_bits(aa):
    bits = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        if (aa >> i) & 1:
            bits[i] = 1
    print(bits)


def filter(raw_signal):
    return scipy.signal.filtfilt(b, a, raw_signal, 1)

def near_dc_lpf(raw_signal):
    filter_signal = []
    for i in range(len(ndlpf_a)):
        filter_signal.append(scipy.signal.filtfilt(ndlpf_b[i], ndlpf_a[i], raw_signal))
    return np.vstack(filter_signal)

def avg_filter_1d(raw_signal, w=5, m="same"):
    win = np.ones(w) / w
    return np.convolve(raw_signal, win, mode=m)

def avg_filter(raw_signal, w=5, m="same"):
    win = np.ones(w)
    filter_signal = np.zeros(raw_signal.shape, dtype=np.float64)
    for i in range(raw_signal.shape[0]):
        filter_signal[i, :] = np.convolve(raw_signal[i, :], win, mode=m) / w
    return filter_signal

def get_phase_diff(raw_signal):
    return raw_signal[:, 0:-1].real * raw_signal[:, 1:].imag - raw_signal[:, 0:-1].imag * raw_signal[:, 1:].real

def get_phase_cum(raw_signal):
    phase_diff = raw_signal[:, 0:-1].real * raw_signal[:, 1:].imag - raw_signal[:, 0:-1].imag * raw_signal[:, 1:].real
    phase_cum = np.cumsum(phase_diff, axis=1)
    return phase_cum

def get_phase_cum_1d(raw_signal):
    phase_diff = raw_signal[0:-1].real * raw_signal[1:].imag - raw_signal[0:-1].imag * raw_signal[1:].real
    phase_cum = np.cumsum(phase_diff)
    return phase_cum

def get_phase(raw_signal):
    angle = np.arctan2(raw_signal.imag, raw_signal.real)
    phase = np.unwrap(angle, axis=1)
    return phase

power_of_2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128]], dtype=np.uint8)

def bits_to_bytes(bits):
    bits = bits.reshape([bits.shape[0], int(bits.shape[1]/8), 8])
    bytes = np.sum(bits * power_of_2, axis=2)
    return bytes
    

def decode_temperature(ori_phase_diff, preamble_idx, native=False, chan=37):
    data_len = 5
    if not native:
        start_len = (1 + 4) * 8 * config.sample_pre_symbol
        seg_len = (1 + 4 + data_len) * 8 * config.sample_pre_symbol
    else:
        start_len = (1 + 4 + 6) * 8 * config.sample_pre_symbol
        seg_len = (1 + 4 + 6 + data_len) * 8 * config.sample_pre_symbol
    phase_diff = np.zeros([ori_phase_diff.shape[0], seg_len-start_len], dtype=np.float32)
    counter = 0
    valid_idx = []
    for i in range(ori_phase_diff.shape[0]):
        end_idx = preamble_idx[i]+seg_len
        if preamble_idx[i] == 0 or end_idx >= ori_phase_diff.shape[1]:
            print(i, "preamble idx error", preamble_idx[i], end_idx, ori_phase_diff.shape[1])
            continue
        phase_diff[counter, :] = ori_phase_diff[i, preamble_idx[i]+start_len:preamble_idx[i]+seg_len]
        valid_idx.append(i)
        counter += 1
    phase_diff = phase_diff[:counter, :]
    
    phase_diff = phase_diff.reshape([phase_diff.shape[0], data_len * 8, config.sample_pre_symbol])
    vote = np.sum(phase_diff, axis=2)
    bits = np.zeros([phase_diff.shape[0], phase_diff.shape[1]], dtype=np.uint8)
    bits[np.where(vote>=0)] = 1
    bytes = bits_to_bytes(bits)
    if not native:
        bytes = bytes ^ scramble_table[chan, :data_len][np.newaxis, :]
    else:
        bytes = bytes ^ scramble_table[chan, 2:2+data_len][np.newaxis, :]
    return bytes

def decode_and_check(phase_diff, preamble_idx, native=False, compare_num=4):
    if preamble_idx+8*config.sample_pre_symbol+compare_num*config.sample_pre_symbol > len(phase_diff):
        return False
    pd_seg = phase_diff[preamble_idx+8*config.sample_pre_symbol: preamble_idx+8*config.sample_pre_symbol+compare_num*config.sample_pre_symbol]
    phase_diff_bit = pd_seg.reshape(-1, config.sample_pre_symbol)
    vote = np.sum(phase_diff_bit, axis=1)
    bits = np.zeros(compare_num, dtype=np.uint8)
    bits[np.where(vote>0)[0]] = 1
    if not native and (bits==target_aa_bit[:compare_num]).all():
        return True
    elif native and (bits==native_target_aa_bit[:compare_num]).all():
        return True
    else:
        return False

# Return index is in the phase difference
# In the raw signal, the index should +1
def find_preamble(phase_diff, native=False, origin_preamble=None, compare_num=4, batch_size=64):
    # The mask only include the last 7 bits of preamble
    # The reason is, the first bit is not exactly same as last 7bits beacuse of the gaussian filter
    preamble_mask = get_preamble_mask(native)
    corrlation = np.zeros([phase_diff.shape[0], phase_diff.shape[1]-preamble_mask.shape[2]+1], dtype=np.float32)

    for i in range(int(np.ceil(phase_diff.shape[0] / batch_size))):
        end_idx = min((i+1)*batch_size, phase_diff.shape[0])
        start_idx = i * batch_size
        if isinstance(phase_diff, np.ndarray):
            phase_diff_tensor = torch.from_numpy(phase_diff[start_idx:end_idx, :]).to(device)
            phase_diff_tensor = phase_diff_tensor.unsqueeze(1).type(torch.float32).to(device)
            out = F.conv1d(phase_diff_tensor, preamble_mask).squeeze(1)
            corrlation[start_idx:end_idx, :] = out.cpu().numpy()
        else:
            phase_diff_tensor = phase_diff[start_idx:end_idx].to(device)
            phase_diff_tensor = phase_diff_tensor.unsqueeze(1).type(torch.float32).to(device)
            out = F.conv1d(phase_diff_tensor, preamble_mask).squeeze(1)
            corrlation[start_idx:end_idx, :] = out.cpu().numpy()
    # Then we start to find the preamble peaks
    preamble_start = np.zeros(phase_diff.shape[0], dtype=np.int32)
    max_arg = np.argsort(corrlation, axis=1)

    for i in range(phase_diff.shape[0]):
        for j in range(-1, -(corrlation.shape[1]+1), -1):
            p_idx = max_arg[i, j]
            if p_idx < config.sample_pre_symbol:
                continue
            p_idx = p_idx - config.sample_pre_symbol
            if decode_and_check(phase_diff[i, :], p_idx, native, compare_num):
                if origin_preamble is not None and np.abs(p_idx - origin_preamble[i]) >= config.sample_pre_symbol:
                    continue

                preamble_start[i] = p_idx  # Since we skip the first one
                break
            if j < -100:
                print("{}th signal has no preamble founded".format(i))
                # plt.figure(figsize=(12, 6))
                # plt.plot(np.cumsum(phase_diff[i, :]))
                # plt.savefig("./errors/not_find_" + str(i) + ".pdf")
                # plt.close()
                break
    # preamble_start = np.argmax(corrlation, axis=1)
    return preamble_start, corrlation

# It seems that there can be a very high peak at the begin and end of the raw signal.
# These peaks is caused by the LPF, since it does not handle the boundaries well
# We just ignore few sample points at very beginning and ending
def normalize(origin_data, zero_center=True, omitted_len=20):
    data = origin_data[:, omitted_len:origin_data.shape[1]-omitted_len]
    if np.iscomplexobj(origin_data):
        max_real = np.max(data.real, axis=1)
        min_real = np.min(data.real, axis=1)
        max_imag = np.max(data.imag, axis=1)
        min_imag = np.min(data.imag, axis=1)
        max_combo = np.hstack([max_real.reshape([-1, 1]), max_imag.reshape([-1, 1])])
        min_combo = np.hstack([min_real.reshape([-1, 1]), min_imag.reshape([-1, 1])])
        max_val = np.max(max_combo, axis=1)
        min_val = np.min(min_combo, axis=1)
    else:
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
    interval = (max_val - min_val).reshape([-1, 1])
    if zero_center:
        return origin_data / interval, interval.reshape([-1])
    else:
        return (origin_data - min_val) / interval, interval.reshape([-1])

def get_cfo(phase, preamble_idx, phase_amp=None):
    phase_slope = np.zeros(phase.shape[0], dtype=np.float64)
    phase_slope_amp = None
    if phase_amp is not None:
        phase_slope_amp = np.zeros(phase.shape[0], dtype=np.float64)
    for i in range(phase.shape[0]):
        if preamble_idx[0] == 0:
            continue
        upper_preamble_bit_idx = np.array([preamble_idx[i]+bit_idx*config.sample_pre_symbol for bit_idx in range(2, 9, 2)], dtype=np.int32)
        lower_preamble_bit_idx = np.array([preamble_idx[i]+bit_idx*config.sample_pre_symbol for bit_idx in range(1, 8, 2)], dtype=np.int32)
        upper_slope, _ = np.polyfit(upper_preamble_bit_idx, phase[i, upper_preamble_bit_idx], 1)
        lower_slope, _ = np.polyfit(lower_preamble_bit_idx, phase[i, lower_preamble_bit_idx], 1)
        phase_slope[i] = (upper_slope + lower_slope) / 2
        if phase_amp is not None:
            amp_upper_slope, _ = np.polyfit(upper_preamble_bit_idx, phase_amp[i, upper_preamble_bit_idx], 1)
            amp_lower_slope, _ = np.polyfit(lower_preamble_bit_idx, phase_amp[i, lower_preamble_bit_idx], 1)
            phase_slope_amp[i] = (amp_upper_slope + amp_lower_slope) / 2
        # if i < 10:
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(phase[i, :])
        #     plt.savefig("./figs/cfo_2" + str(i) + ".pdf")
        #     plt.close()
        
    # for now, the cfo is the phase difference per sample point, translate it into frequency

    cfo = phase_slope * config.sample_rate / 2 / np.pi
    return cfo, phase_slope, phase_slope_amp
    
def cfo_compensate(signal, cfo):
    if signal.ndim == 1:
        signal = signal.reshape([1, -1])
        cfo = np.array([cfo])

    t = np.arange(0, signal.shape[1]) * (1 / config.sample_rate)
    time = np.array([t for _ in range(signal.shape[0])])
    cfo = cfo.reshape([-1, 1])
    freq_dev = 2 * np.pi * (-cfo) * time
    freq_dev_signal = np.exp(1j * freq_dev)
    return signal * freq_dev_signal

def cfo_compensate_1d(signal, cfo):
    t = np.arange(0, len(signal)) / config.sample_rate
    freq_dev = 2 * np.pi * (-cfo) * t
    return signal * np.exp(1j * freq_dev)

def frequency_deviation(signal, deviation):
    t = np.arange(0, signal.shape[1]) / config.sample_rate
    time = np.array([t for _ in range(signal.shape[0])])
    freq_dev = 2 * np.pi * deviation * time
    return signal * np.exp(1j * freq_dev)

def add_noise(raw_signal, noise_power):
    iq_noise = noise_power * np.random.normal(0, 1, (raw_signal.shape[0], raw_signal.shape[1], 2))
    noise_signal = np.zeros(raw_signal.shape, dtype=np.complex64)
    noise_signal[:, :].real = raw_signal[:, :].real + iq_noise[:, :, 0]
    noise_signal[:, :].imag = raw_signal[:, :].imag + iq_noise[:, :, 1]
    return noise_signal



# To gerneate a gaussian FIR for pulse shapping, same as gaussdesign in Matlab
def gaussdesign(bt=0.5, span=3, sps=20):
    filtLen = span * sps + 1
    t = np.linspace(-span/2, span/2, filtLen)
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi)/alpha) * np.exp(-(t*np.pi/alpha)**2); 
    h = h / np.sum(h)
    return h

def gfsk_ref_signal(preamble=None, postfix=None, freq_dev=170e3, init_phase=0):
    g_filter = gaussdesign(sps=config.sample_pre_symbol)

    if preamble is None:
        preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    if postfix is None:
        postfix = np.array([0, 0, 0], dtype=np.int8)
    
    data = np.concatenate([preamble, postfix])
    # The GFSK first up-sample the data and turn it into NRZ code
    up_sampling = np.zeros((len(data)) * config.sample_pre_symbol, dtype=np.float64)
    for i in range(len(data)):
        up_sampling[i*config.sample_pre_symbol: (i+1)*config.sample_pre_symbol] = data[i] * 2 - 1
    
    # Use gaussian pulse shape filter 
    gamma_gfsk = scipy.signal.lfilter(g_filter, np.array([1], dtype=np.float64), up_sampling)
    gfsk_phase = np.zeros(len(gamma_gfsk))
    gfsk_phase[1:] = 2 * np.pi * freq_dev * (1 / config.sample_rate) * np.cumsum((gamma_gfsk[:-1] + gamma_gfsk[1:]) / 2 )
    gfsk_phase += init_phase
    gfsk_phase = gfsk_phase[int(1.5*config.sample_pre_symbol):int(9.5*config.sample_pre_symbol)]
    gfsk_phase = gfsk_phase[:] - gfsk_phase[0]
    return gfsk_phase

# The raw_signal_seg is combined with config.feat_len
def cal_SNR(raw_signal_seg, noise_seg, scale):
    preamble_raw = raw_signal_seg[:, config.feat_len:]
    phase = get_phase(preamble_raw)
    phase_diff_amp = get_phase_diff(preamble_raw)
    phase_diff_std = np.sin(phase[:, 1:] - phase[:, 0:-1])
    signal_power = np.average(phase_diff_amp / phase_diff_std, axis=1) / 2 * (scale ** 2)
    noise_power = np.zeros(raw_signal_seg.shape[0], dtype=np.float64)
    for i in range(raw_signal_seg.shape[0]):
        noise_power[i] = np.mean(np.abs(noise_seg[i]) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def load_and_get_cfo(file, native):
    data = np.load(file)
    raw_data = data["arr_0"]
    comp_signal = filter(raw_data)
    comp_phase_diff = get_phase_diff(comp_signal)
    preamble_idx, _ = find_preamble(comp_phase_diff, native=native, compare_num=32)
    phase = get_phase(comp_signal)
    cfo, _, _ = get_cfo(phase, preamble_idx)
    cfo = cfo[np.where(preamble_idx!=0)]
    return cfo



    # gfsk_phase = gfsk_ref_signal(freq_dev=freq_dev)
    # gfsk_phase_point = gfsk_phase[[config.sample_pre_symbol * i for i in range(1, 8)]]
    # signal_phase = get_phase(raw_signal_seg[:, config.feat_len:])
    # signal_phase_points = signal_phase[:, [config.sample_pre_symbol * i for i in range(1, 8)]]
    # phase_difference = np.average(signal_phase_points - gfsk_phase_point, axis=1)
    # for i in range(10):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(gfsk_phase[:], label="gfsk")
    #     plt.plot(signal_phase[i, :], label="signal")    
    #     combo = np.concatenate([gfsk_phase[:], signal_phase[i, :]]) 
    #     for j in range(8):
    #         plt.plot([j * config.sample_pre_symbol, j * config.sample_pre_symbol], [np.max(combo), np.min(combo)])
    #     plt.legend()
    #     plt.savefig("./test_figs/snr_" + str(i) + ".pdf")
    #     plt.close()   
    # for i in range(raw_signal_seg.shape[0]):
    #     ref_signal = np.exp(1j * (phase_difference[i] + gfsk_phase))
    #     ref 


if __name__ == "__main__":
    get_preamble_bits(nrf_aa)
    # gfsk_phase = gfsk_ref_signal(init_phase=0)
    # sym_start_phase = gfsk_phase[[20 * i for i in range(8)]]
    # print(sym_start_phase)
    # data = np.load("./raw_data/nrf52840_raw/d=1m_ttt_1_nodeid_" + str(1) + ".npz")["arr_0"]
    # data = filter(data)
    # phase_diff = get_phase_diff(data)
    # phase_cum = get_phase_cum(data)
    # b = np.array([i*config.sample_pre_symbol for i in range(8)])

    # p_idx, corrlation = find_preamble(phase_diff)
    # for i in range(10):
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(2, 1, 1)
    #     plt.plot(phase_cum[i, :1000])
    #     plt.plot(p_idx[i]+b, phase_cum[i, p_idx[i]+b])
    #     plt.xlim([0, 1000])
    #     plt.subplot(2, 1, 2)
    #     plt.plot(corrlation[i, :1000])
    #     plt.xlim([0, phase_cum.shape[1]])
    #     print(p_idx[i])
    #     plt.savefig("./test_figs/preamble_" + str(i) + ".pdf")
    #     plt.close()

    # phase = get_phase_cum(signal.reshape([1, -1]))[0, :]
    # plt.figure(figsize=(12, 12))
    # plt.subplot(2, 1, 1)
    # plt.plot(signal.real)
    # plt.plot(signal.imag)
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(1, 160), phase[:])
    # plt.plot(gfsk_phase)
    # plt.xticks(np.arange(0, 160, 20))
    # plt.savefig("./test_figs/gfsk_sim.pdf")
    # plt.close()


