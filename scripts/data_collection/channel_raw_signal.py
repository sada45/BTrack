import uhd
import numpy as np
import config
import threading
import time 
from scramble_table import *
import cv2
from node_ctrl import node_ctrl
from scipy import signal
from scramble_table import scramble_table
from multiprocessing import Process, Queue, set_start_method

# buffer = ringbuffer(config.num_samps, np.complex64)
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
b, a = signal.butter(8, 2e6 / config.sample_rate, "lowpass")

# [0xee, 0xf1, 0x4b, 0xbf, 0x71, 0x3c]

esp_mac = [[90, 230, 75, 191, 113, 60]]

mac_bits_list = None

# /* RF center frequency for each channel index (offset from 2400 MHz) */
# static const uint8_t g_ble_phy_chan_freq[BLE_PHY_NUM_CHANS] = {
#      4,  6,  8, 10, 12, 14, 16, 18, 20, 22, /* 0-9 */
#     24, 28, 30, 32, 34, 36, 38, 40, 42, 44, /* 10-19 */
#     46, 48, 50, 52, 54, 56, 58, 60, 62, 64, /* 20-29 */
#     66, 68, 70, 72, 74, 76, 78,  2, 26, 80, /* 30-39 */
# };

def byte_to_bits(b):
    bits = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        if (b >> i) & 1:
            bits[i] = 1
        else:
            bits[i] = 0
    return bits

def mac_channel_mask(mac, chan):
    bits = np.zeros(48, dtype=np.uint8)
    for i in range(6):
        mac_byte = mac[i] ^ scramble_table[chan][2+i]
        bits[i*8: i*8+8] = byte_to_bits(mac_byte)
    return bits

def mac_to_bits(macs):
    mac_bits = []
    for i in range(len(macs)):
        bits = np.zeros(32, dtype=np.uint8)
        for j in range(4):
            bits[j*8: (j+1)*8] = byte_to_bits(macs[i][j])
        mac_bits.append(bits)
    return mac_bits


def get_freq_with_chan_num(chan):
    if chan == 37:
        return 2402000000
    elif chan == 38:
        return 2426000000
    elif chan == 39:
        return 2480000000
    elif chan >= 0 and chan <= 10:
        return 2404000000 + chan * 2000000
    elif chan >= 11 and chan <= 36:
        return 2428000000 + (chan - 11) * 2000000
    else:
        return 0xffffffffffffffff

def get_aa_bits(aa_byte):
    aa_bits = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        if (aa_byte >> i) & 1:
            aa_bits[i] = 1
    return aa_bits

def search_sequence(bits, target_bits):
    res = cv2.matchTemplate(bits, target_bits, cv2.TM_SQDIFF)
    idx = np.where(res==0)[0]
    if len(idx) > 0:
        return idx
    else:
        return None

def raw_signal_segment(data, raw_signal, timestamp, cfo, target_mac_bit=None):
    t = np.arange(0, len(data)) / config.sample_rate
    phase_compensate = np.exp(1j * 2 * np.pi * t * (-cfo))
    data = data * phase_compensate
    i0 = np.real(data[:-1])
    q0 = np.imag(data[:-1])
    i1 = np.real(data[1:])
    q1 = np.imag(data[1:])
    phase_diff = i0 * q1 - i1 * q0
    phase_len = len(phase_diff)
    bits = np.zeros(int(np.ceil(phase_len / config.sample_pre_symbol)), dtype=np.uint8)
    segment_len = 64
    if target_mac_bit is not None:
        segment_len += 80  # recorde the MAC address
        # print(target_mac_bit)
    for i in range(config.sample_pre_symbol):
        bits[:] = 0
        sample_len = (phase_len - i) // config.sample_pre_symbol
        p = phase_diff[i: i+sample_len*config.sample_pre_symbol].reshape(-1, config.sample_pre_symbol)
        vote = np.sum(p, 1)
        bits[np.where(vote>0)[0]] = 1
        preamble_idx = search_sequence(bits, target_aa_bit)
        if preamble_idx is not None:
            for j in range(len(preamble_idx)):
                idx = (preamble_idx[j] - 50) * config.sample_pre_symbol
                end_idx = idx + config.sample_pre_symbol * segment_len + config.sample_pre_symbol
                if target_mac_bit is not None:
                    mac_idx = preamble_idx[j] + 48 + 8
                    if mac_idx + 48 < len(bits):
                        if not (bits[mac_idx:mac_idx+48] == target_mac_bit).all():
                            continue
                    else:
                        continue

                # empty_idx = end_idx + 10 * config.sample_pre_symbol
                # empty_end_idx = empty_idx + config.sample_pre_symbol * segment_len + config.sample_pre_symbol
                # if idx >=0 and empty_end_idx < len(data):
                if idx >= 0 and end_idx < len(raw_signal):
                    # bits_data = bits[preamble_idx[j]: preamble_idx[j] + 166 * 8].reshape(-1, 8)
                    # bytes_data = np.sum(bits_data * power_of_2, 1, dtype=np.uint8)
                    # bytes_data[4: 6] ^= scramble_table[8][:2]
                    # bytes_data = bytearray(bytes_data)
                    # print(bytes_data.hex())
                    raw_data = raw_signal[idx: end_idx]
                    # empty_raw_data = data[empty_idx: empty_end_idx]
                    # print(idx, end_idx)
                    # return raw_data, empty_raw_data, timestamp[0] + idx * (1 / config.sample_rate)
                    return raw_data.copy(), timestamp[0] + idx * (1 / config.sample_rate)
                else:
                    print("sample too short", i, idx, end_idx)
    print("not find")
    return None, -1

class usrp_control(threading.Thread):
    def __init__(self, usrp_type, chan, prefix, mac=None):
        threading.Thread.__init__(self)
        self.n1 = uhd.usrp.MultiUSRP("type=" + usrp_type)
        self.n1.set_rx_rate(config.sample_rate, 0)
        self.n1.set_rx_freq(uhd.libpyuhd.types.tune_request(get_freq_with_chan_num(chan)), 0)
        self.n1.set_rx_gain(config.gain, 0)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.metadata = uhd.types.RXMetadata()
        self.streamer = self.n1.get_rx_stream(st_args)
        self.recv_buffer = np.zeros((1, config.num_samples_each), dtype=np.complex64)
        self._is_streaming = True
        self.sample = np.zeros((config.extra_repeat_times, config.num_samples), dtype=np.complex64)
        self.time_stamp = np.zeros((config.extra_repeat_times, config.num_samples // config.num_samples_each), dtype=np.float64)
        self.stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        self.stream_cmd.stream_now = True
        self.stream_cmd.num_samps = config.num_samples_each
        self.raw_data_segments = []
        self.raw_data_timestamp = []
        self.raw_data_index = []
        self.raw_empty_data_segments = []
        # two array to store
        self.raw_signal_arr = None
        self.time_stamp_arr = None 
        self.prefix = prefix + "_chan=" + str(chan)
        if mac is not None:
            self.mac_bits = mac_channel_mask(mac, chan)
        else:
            self.mac_bits = None

    def start_stream(self, sample_num=config.extra_repeat_times):
    # Set up the stream and receive buffer
        # Start Stream
        print("sample_num=", sample_num)
        self.streamer.issue_stream_cmd(self.stream_cmd)
        time_diff = (time.time_ns() // 1e9) - self.n1.get_time_now().get_real_secs()
        sample_index = 0
        st = time.time()
        last_time_diff = 0
        while sample_index < sample_num:
            for i in range(config.num_samples // config.num_samples_each):
                # Receive Samples
                s_len = self.streamer.recv(self.recv_buffer, self.metadata)
                start_timestamp = self.metadata.time_spec.get_real_secs()
                self.time_stamp[sample_index, i] = start_timestamp + time_diff
                # print(sample_index, start_timestamp-last_time_diff)
                last_time_diff = start_timestamp
                self.sample[sample_index, config.num_samples_each*i: config.num_samples_each*(i+1)] = self.recv_buffer[0]
            sample_index += 1
        print("data collection time", time.time()-st)
            # raw_data, timestamp = raw_signal_segment(self.sample[sample_index, :], self.time_stamp[sample_index, :])
            # if raw_data is not None:
            #     self.raw_data_segments.append(raw_data)
            #     self.raw_data_timestamp.append(timestamp)
            #     # print(self.time_stamp[sample_index, :])
            #     print(self.time_stamp[sample_index, 1:] - self.time_stamp[sample_index, :-1])
            #     sample_index += 1

    def stop_stream(self, sample_num=config.extra_repeat_times):
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
        # np.savez("all_raw_signal.npz", self.sample)
        
        print("start processing")
        for i in range(sample_num):
            filtered_signal = signal.filtfilt(b, a, self.sample[i, :])
            raw_signal, signal_time_stamp = raw_signal_segment(filtered_signal, self.sample[i, :], self.time_stamp[i, :], 0, self.mac_bits)
            if raw_signal is not None:
                self.raw_data_segments.append(raw_signal)
                self.raw_data_timestamp.append(signal_time_stamp)
                self.raw_data_index.append(i)
                print(i, "/", len(self.raw_data_segments))
                if len(self.raw_data_segments) == config.repeat_times:
                    return
                # self.raw_empty_data_segments.append(empty_raw_signal)
            else:
                print(i)


    def run(self):
        # while self._is_streaming:
        self.start_stream()
        self.stop_stream()

        while True:
            last_remain_num = config.extra_repeat_times
            while len(self.raw_data_segments) < config.repeat_times:
                if len(self.raw_data_segments) == 0:
                    self.start_stream(config.extra_repeat_times)
                    self.stop_stream(config.extra_repeat_times)
                else:
                    print("retrans", len(self.raw_data_segments), last_remain_num, config.repeat_times - len(self.raw_data_segments))
                    sample_num = int(min(np.ceil(1.5 * (config.repeat_times - len(self.raw_data_segments))), config.repeat_times))
                    if config.repeat_times - len(self.raw_data_segments) == last_remain_num:
                        sample_num = min(int(np.ceil(1.5 * sample_num)), config.extra_repeat_times)
                    self.start_stream(sample_num)
                    self.stop_stream(sample_num)
                    last_remain_num = config.repeat_times - len(self.raw_data_segments)
            self.raw_signal_arr = np.array(self.raw_data_segments)
            self.time_stamp_arr = np.array(self.raw_data_timestamp)
            if self.raw_signal_arr.ndim != 2:
                for i in range(len(self.raw_data_segments)):
                    print(i, self.raw_data_segments[i, :].shape)
                self.raw_data_segments = []
                self.raw_data_timestamp = []
                self.raw_data_index = []
            else:
                break
        np.savez(self.prefix + ".npz", self.raw_signal_arr, self.time_stamp_arr)

    def stop(self):
        self._is_streaming = False

def get_chan_raw():
    # for nodeid in range(1, 9):
    global target_aa_bit 
    target_aa_bit = np.concatenate([preamble, get_aa_bits(0x12345678)])
    for nodeid in [1]:
        nc = node_ctrl(nodeid)
        for chan in [37]:
            while True:
                res = 0
                res += nc.stop()
                res += nc.set_chan(chan)
                res += nc.start()
                if res < 0:
                    print("chan ", chan, "error, retrying")
                    time.sleep(1)
                else:
                    break
            for i in range(1, 2):
                print("./raw_data/single_chan/" + "chan=" + str(chan) + "_nodeid=" + str(nodeid))
                # 180048
                # 8001044
                rx_thread = usrp_control("b200, serial=8001044", chan, "./raw_data/nrf52840_raw/d=0_ttt_" + str(i+1) + "_nodeid_" + str(nodeid))
                rx_thread.daemon = False
                rx_thread.start()
                rx_thread.join()
                del rx_thread
        res += nc.stop()
        del nc

def get_chan_raw_esp():
    global target_aa_bit 
    global mac_bits_list
    mac_bits_list = mac_to_bits(esp_mac)
    print(mac_bits_list)
    # target_aa_bit = mac_bits_list[0]
    target_aa_bit = np.concatenate([preamble, get_aa_bits(0x8e89bed6)])

    for nodeid in range(1):
        # e = esp_control.esp_controller("/dev/ttyUSB0")
        # while True:
        #     if e.start() == 0:
        #         break
        for chan in [37]:
            rx_thread = usrp_control("b200, serial=8001044", chan, "./raw_data/esp_raw/test=0_2_espid_" + str(nodeid), esp_mac[nodeid])
            rx_thread.daemon = False
            rx_thread.start()
            rx_thread.join()
            del rx_thread
        # while True:
        #     if e.stop() == 0:
        #         print("stopped")
        #         break

# get_chan_raw_esp()
get_chan_raw()