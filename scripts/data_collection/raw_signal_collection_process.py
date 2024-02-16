import uhd
import numpy as np
import config
import threading
import time 
from scramble_table import *
import cv2
import node_ctrl 
from scipy import signal
from scramble_table import scramble_table
import threading
from multiprocessing import Process, Semaphore, Pipe, Event, Queue

# buffer = ringbuffer(config.num_samps, np.complex64)
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
target_preamble_aa_bit = np.array([0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
upper_bound_idx = np.array([idx*config.sample_pre_symbol for idx in range(2, 9, 2)], dtype=np.int32)
lower_bound_idx = np.array([idx*config.sample_pre_symbol for idx in range(1, 9, 2)], dtype=np.int32)

b, a = signal.butter(8, 2e6 / config.sample_rate, "lowpass")

# [0xee, 0xf1, 0x4b, 0xbf, 0x71, 0x3c]

esp_mac = [[0xee, 0xf1, 0x4b, 0xbf, 0x71, 0x3c]]

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

def cal_phase(data):
    angle = np.arctan2(data.imag, data.real)
    return np.unwrap(angle)

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
    feature_size = 30
    segment_len = feature_size + 8 + 32 + 16 + 48 + 10

    for i in range(config.sample_pre_symbol):
        bits[:] = 0
        sample_len = (phase_len - i) // config.sample_pre_symbol
        p = phase_diff[i: i+sample_len*config.sample_pre_symbol].reshape(-1, config.sample_pre_symbol)
        vote = np.sum(p, 1)
        bits[np.where(vote>0)[0]] = 1
        preamble_idx = search_sequence(bits, target_preamble_aa_bit)
        if preamble_idx is not None:
            for j in range(len(preamble_idx)):
                idx = (preamble_idx[j] - feature_size) * config.sample_pre_symbol
                end_idx = idx + config.sample_pre_symbol * segment_len + config.sample_pre_symbol + 1
                if target_mac_bit is not None:
                    mac_idx = preamble_idx[j] + 48
                    if mac_idx + 48 < len(bits):
                        if not (bits[mac_idx:mac_idx+48] == target_mac_bit).all():
                            continue
                    else:
                        continue
                if idx >= 0 and end_idx < len(raw_signal):
                    return raw_signal[idx: end_idx].copy(), timestamp[0] + idx * (1 / config.sample_rate)
                else:
                    print("sample too short", i, idx, end_idx)

    print("not find")
    return None, -1

class usrp_control(threading.Thread):
    def __init__(self, usrp_type, chan, prefix, sema, continue_sema, total_sema, raw_signal, time_stamp, mac=None):
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
        self._is_terminted = False
        self.sample = raw_signal
        # np.ndarray(shape=(2 * config.sample_pre_symbol, config.num_samples), dtype=np.complex64, buffer=shm_raw_signal.buf)
        # np.zeros([config.repeat_times, config.num_samples], dtype=np.complex64)
        self.time_stamp = time_stamp
        # np.ndarray(shape=(config.repe, config.num_samples // config.num_samples_each), dtype=np.float64, buffer=shm_time_stamp.buf)
        # np.zeros([config.repeat_times, config.num_samples // config.num_samples_each], dtype=np.float64)
        self.stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        self.stream_cmd.stream_now = True
        self.stream_cmd.num_samps = config.num_samples_each
        # two array to store
        self.raw_signal_arr = None
        self.time_stamp_arr = None 
        self.prefix = prefix + "_chan=" + str(chan)
        if mac is not None:
            self.mac_bits = mac_channel_mask(mac, chan)
        else:
            self.mac_bits = None
        self.sema = sema
        self.continue_sema = continue_sema
        self.total_sema = total_sema

    def stop_stream(self):
        print("USRP stop stream")
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
    
    def start_stream(self, sample_num=config.repeat_times):
    # Set up the stream and receive buffer
        # Start Stream
        self.streamer.issue_stream_cmd(self.stream_cmd)
        time_diff = (time.time_ns() // 1e9) - self.n1.get_time_now().get_real_secs()
        sample_index = 0

        # while self._is_streaming:
        while True :
            if not self._is_streaming or (sample_index != 0 and sample_index % config.extra_repeat_times == 0):
                self.stop_stream()
                self.continue_sema.acquire()
                if self._is_terminted:
                    return
                self.streamer.issue_stream_cmd(self.stream_cmd)
            for i in range(config.num_samples // config.num_samples_each):
                # Receive Samples
                s_len = self.streamer.recv(self.recv_buffer, self.metadata)
                start_timestamp = self.metadata.time_spec.get_real_secs()
                self.time_stamp[sample_index % config.extra_repeat_times, i] = start_timestamp + time_diff
                self.sample[sample_index % config.extra_repeat_times, config.num_samples_each*i: config.num_samples_each*(i+1)] = self.recv_buffer[0]
                # print(np.max(self.recv_buffer[0, :].real))
            sample_index += 1
            # print(time.time() - st)
            # if sample_index == 64:
            #     print("64 time", time.time() - st)5
            # print("sample index", sample_index)
            # with self.lock:
            self.sema.release()
            self.total_sema.release()

    def run(self):
        # while self._is_streaming:
        self.start_stream()
        self.stop_stream()

    def stop(self):
        self._is_streaming = False
        self.continue_sema.release()
    
    def continue_rx(self):
        self._is_streaming = True
        self.continue_sema.release()

    def terminate(self):
        self._is_streaming = False
        self._is_terminted = True
        self.continue_sema.release()
    
    def __del__(self):
        print("USRP stop streamming")
        self.stop_stream()

def data_processing_process(recv_p, q, continue_sema, done_sema, total_sema, cfo, mac):
    raw_signal_segs = []
    time_stamps = []
    total_num = 0

    print("Start processing")
    while len(raw_signal_segs) < config.repeat_times:
        raw_signal, timestamp = recv_p.recv()
        total_num += 1
        filtered_data = signal.filtfilt(b, a, raw_signal)
        raw_signal_seg, time_stamp = raw_signal_segment(filtered_data, raw_signal, timestamp, cfo, target_mac_bit=mac)
        if raw_signal_seg is not None:
            raw_signal_segs.append(raw_signal_seg)
            time_stamps.append(time_stamp)
            print(total_num, len(raw_signal_segs))
        if total_num % config.extra_repeat_times == 0:
            print("release")
            continue_sema.release()
    
    q.put([np.array(raw_signal_segs), np.array(time_stamps)])
    done_sema.release()
    total_sema.release()
    recv_p.close()

def collect_raw_data(chan, cfo, prefix, mac=None) :
    raw_signal = np.zeros(shape=(config.extra_repeat_times, config.num_samples), dtype=np.complex64)
    timestamp = np.zeros(shape=(config.extra_repeat_times, config.num_samples // config.num_samples_each), dtype=np.float64)

    sem = Semaphore(0)
    thread_continue_sema = Semaphore(0)
    done_sema = Semaphore(0)
    total_sema = Semaphore(0)
    send_p, recv_p = Pipe()
    segment_idx = 0
    q = Queue()
    # 8001044
    # 8001926
    dp = Process(target=data_processing_process, args=(recv_p, q, thread_continue_sema, done_sema, total_sema, cfo, mac,))
    dp.start()
    ut = usrp_control("b200, serial=8001044", chan, prefix, sem, thread_continue_sema, total_sema, raw_signal, timestamp)
    ut.start()
    while True:
        total_sema.acquire()
        sem_res = sem.acquire(block=False)
        if sem_res:
            send_p.send([raw_signal[segment_idx % config.extra_repeat_times, :], timestamp[segment_idx % config.extra_repeat_times, :]])
            segment_idx += 1
        else:
            done_res = done_sema.acquire(block=False)
            if done_res:
                ut.terminate()
                ut.join()
                del ut
                dp.join()
                send_p.close()
                return q.get()
            else:
                send_p.close()
                raise Exception("Unknow event")

if __name__ == "__main__":
    for nodeid in [8]:
        # nc = node_ctrl.node_ctrl(nodeid)
        for chan in [37]:
            # while True:
            #     res = 0
            #     res += nc.stop()
            #     res += nc.set_chan(chan)
            #     res += nc.start()
            #     if res < 0:
            #         print("chan ", chan, "error, retrying")
            #         time.sleep(1)
            #     else:
            #         break
            for i in range(1):
                print("./raw_data/single_chan/" + "chan=" + str(chan) + "_nodeid=" + str(nodeid))
                # 180048
                # 8001044
                prefix = "./raw_data/nrf52840_raw/d=5m_ttt_" + str(i+1) + "_nodeid_" + str(nodeid)
                raw_signal, _ = collect_raw_data(chan, node_ctrl.cfo_list[nodeid], prefix)
                # np.savez(prefix + ".npz", raw_signal)
        # res += nc.stop()
        # del nc




