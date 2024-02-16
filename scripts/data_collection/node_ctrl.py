import pylink
import time
import numpy as np

# This is the typical CFO for the nodes we have
# If you want to use this code, you should first measure the CFO in high-SNR environment and use the `get_cfo()` in feature_extractor.py to get the CFO
cfo_list = [0, 109501.26941937912,-17247.676310789888,110946.54895685444,107441.501821522,104336.27827251791,6259.23345239831,110678.55929630865,-21286.480752017218]

class node_ctrl():
    def __init__(self, sn):
        lib = pylink.library.Library('/home/sada45/JLink/libjlinkarm.so')
        self.jlink = pylink.JLink(lib)
        self.jlink.open(sn)
        self.jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
        self.jlink.connect("nRF52840_xxAA")
        self.adv_len = 31
        if self.jlink.connected():
            self.jlink.reset(0, False)
            self.jlink.rtt_start()
    
    def write(self, writedata):
        writeindex = 0
        while writeindex < len(writedata):
            bytes_written = self.jlink.rtt_write(0, writedata[writeindex:])
            writeindex = writeindex + bytes_written
            time.sleep(0.01)

    def check_resp(self):
        read_data = ""
        for _ in range(10):
            read_data = read_data.join([chr(c) for c in self.jlink.rtt_read(0, 4096)])
            print(read_data)
            if read_data.find("fail") != -1:
                # print("errors")
                return -1
            elif read_data.find("done") != -1:
                # print("no error")
                return 0
            time.sleep(0.1)
        print("time out")
        return -1
    
    def stop(self):
        writedata = list(bytearray("stop", "utf-8") + b'\r\n')
        self.write(writedata)
        return self.check_resp()

    def start(self):
        writedata = list(bytearray("start", "utf-8") + b'\r\n')
        self.write(writedata)
        return self.check_resp()

    def set_chan(self, chan):
        writedata = list(bytearray("chc " + str(chan), "utf-8") + b'\r\n')
        self.write(writedata)
        return self.check_resp()
    
    def set_data(self, cmd):
        writedata = list(bytearray(cmd, 'utf-8') + b'\r\n')
        self.write(writedata)
        return self.check_resp()

    def set_extend_factor(self, extf):
        writedata = list(bytearray("che " + str(extf), "utf-8") + b'\r\n')
        self.write(writedata)
        return self.check_resp()
    
    
    def set_extf_nmax(self, extf, n_max):
        writedata = list(bytearray("chen " + str(extf) + " " + str(n_max), "utf-8") + b'\r\n')
        self.write(writedata)
        return self.check_resp()
    
    def change_adv_data(self):
        rand_bytes = np.random.randint(256, size=(self.adv_len), dtype=np.uint8)
        cmd_str = "chd " + str(self.adv_len) + " "
        for b in rand_bytes:
            cmd_str += str(b) + ' '
        self.set_data(cmd_str)
        if self.check_resp() == 0:
            return rand_bytes
        else:
            return None
    
    def __del__(self):
        self.jlink.rtt_stop()
        self.jlink.close()