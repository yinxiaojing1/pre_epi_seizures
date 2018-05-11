"""
Listen to serial, return most recent numeric values
Lots of help from here:
http://stackoverflow.com/questions/1093598/pyserial-how-to-read-last-line-sent-from-serial-device
"""
from threading import Thread
import time

import pluxv2
from pluxv2 import *

import struct
import time
import sys

last_received = 0
def receiving(device):
    global last_received
    frames = []
    device.start()
    ch = 4
    while True:
        try:
            last_received = float(device.frames(1)['Frames'][:,ch][0])
            time.sleep(0.001)
        except Exception as e:
            pass
        
class PluxData(object):
    def __init__(self, mac='test', SamplingRate=1000.):
        try:
            self.device = BioPluxDevice(mac, SamplingRate)
            # self.device.start()
        except Exception as e:
            print e
            self.device = None
        else:
            self.t = Thread(target=receiving, args=(self.device,)).start()
        
    def next(self):
        if not self.device:
            return 100 #return anything so we can test when Bioplux isn't connected
        return last_received
    # def __del__(self):
        # if self.device:
            # self.device.stop()
            # self.device.disconnect()

if __name__=='__main__':
    s = PluxData()
    for i in range(1000):
        print s.next(),
        # time.sleep(.001)
    try:
        s.device.stop()
        s.device.disconnect()
    except Exception as e:   
        pass
        