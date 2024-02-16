import numpy as np
import scripts.fext.utils as utils
import feature_extractor as fext

for nodeid in range(1, 9):
    data = np.load("./raw_data/nrf52840_raw/d=0_nodeid_" + str(nodeid) + "_chan=37.npz")["arr_0"]
    data = utils.filter(data)
    phase = utils.get_phase(data)
    phase_diff = utils.get_phase_diff(data)
    phase_cum = utils.get_phase_cum(data)
    preamble_idx, _ = utils.find_preamble(phase_diff, False)
    cfo, slope, _ = fext.get_cfo(phase, preamble_idx)
    print(np.average(cfo), end=",")