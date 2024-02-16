# sample_rate = 16e6
# sample_pre_symbol = int(sample_rate // 1e6)
# chan = 10
# access_address = 0x8E89BED6
# crc_init = 0x555555
# gain = 40
# num_samples = int(20e5)
# num_samples_each = int(5e5)
# repeat_times = 10

# # sample rate = 4M
# sample_rate = 4e6
# sample_pre_symbol = int(sample_rate // 1e6)
# chan = 0
# access_address = 0x8E89BED6
# crc_init = 0x555555
# gain = 40
# num_samples = int(45e4)
# num_samples_each = int(45e4)
# repeat_times = 64

sample_rate = 20e6
sample_pre_symbol = int(sample_rate // 1e6)
access_address = 0x8E89BED6
crc_init = 0x555555
gain = 40
num_samples = int(25e5)
num_samples_each = int(5e5)
repeat_times = 10
extra_repeat_times = int(1.5 * repeat_times)
# data preprocessing config
stft_window_size = 4  # in unit config.sample_pre_symbol
stft_pad_size = 3 # number of 0s for padding

file_batch_size = 1
# model config
batch_size = 64
seed = 11
chan_signal_num = 64
loss_scale = 32