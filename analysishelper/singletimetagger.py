import numpy as np
import time
#import os
#import copy
import zlib
from zmqhelper import Client
import re


class TimeTagger():
    """
    Simple class to connect to a single timetagger.
    """

    def __init__(self, ip, port='50000', channelmap=None):
        self.ip = ip
        self.port = port
        self.connect()
        self.dataType = np.dtype([('ch', np.uint8), ('ttag', np.uint64)])
        self.filename = ''
        # print(self.sendMessage('commands'))

    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass

    def sendMessage(self, message):
        ret = self.socket.send_message(message)
        return(ret)

    def start_server(self):
        response = self.sendMessage('start')
        if response is not None:
            return response
        else:
            return("Error communicating with server")

    def stop_server(self):
        response = self.sendMessage('done')
        if response is not None:
            return response
        else:
            return("Error communicating with server")

    def start_logging_to_file(self, filepostfix=''):
        response = self.sendMessage('log on ' + filepostfix)
        filename = ''
        try:
            filename = re.sub('\\nAck\n$', '', response)
        except Exception:
            filename = ''
        self.filename = filename
        return filename

    def stop_logging_to_file(self):
        response = self.sendMessage('log off')
        filename = ''
        try:
            filename = re.sub('\\nAck\n$', '', response)
        except Exception:
            filename = ''
        self.filename = filename
        return filename

    def get_data_from_file(self, fname, numTtags):
        print("file to stream: ", fname)
        response = self.sendMessage('start file %s' % fname)
        time.sleep(1)
        binData = self.sendMessage('stream %s' % numTtags)
        data = self.convert_data(binData)

        return data

    def connect(self):
        self.socket = Client(self.ip, self.port)

    def use_10_MHZ(self, use10mhz):
        use10mhz = int(use10mhz)
        msg = self.sendMessage('use10mhz '+str(use10mhz))
        return msg

    def set_ch_level(self, ch, level):
        msg = 'setchlevel '
        msg += str(ch)+' '
        msg += str(level)
        retmsg = self.sendMessage(msg)
        return retmsg

    def calibrate(self):
        self.sendMessage('stopbuff')
        time.sleep(1)
        msg = self.sendMessage('calibrate')
        self.sendMessage('start')
        return msg

    def convert_data(self, binData):
        try:
            binData = zlib.decompress(binData)
        except Exception:
            pass

        data = None

        try:
            data = np.frombuffer(binData, dtype=self.dataType)
        except Exception:
            try:
                if binData == 'Timeout':
                    print("Timeout fetching data")
            except Exception:
                print('fail')
            data = None
        return data

    def stream_server(self, dt=1):
        binData = self.sendMessage('stream %f' % dt)
        data = self.convert_data(binData)
        return(data)

    def get_stats(self, dt=0.5):

        msg = ("getcounts " + str(dt))
        ret = self.sendMessage(msg)
        ret = ret
        if ret == 'Timeout':
            counts = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            ret = (ret.replace("[", '').replace(']', ''))
            counts = [int(float(s)) for s in ret.split(',')]

        return counts


if __name__ == '__main__':

    print('connecting')
    ip = 'tcp://132.163.53.24'
    port = 50000
    chns = ''
    tt = TimeTagger(ip, port, chns)
    stats = tt.get_stats()
    print(stats)
    data = tt.stream_server(dt=0.5)
    print(data)
