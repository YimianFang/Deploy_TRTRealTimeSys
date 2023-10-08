import os
os.environ['OPENCV_IO_ENABLE_JASPER']= 'True'
import cv2
import numpy as np
from flt_trtexec_utils import *
import argparse
import threading
from queue import Queue
import time, struct
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default="192.168.1.188:50051")
args = parser.parse_args()

E_engine_file_path = "flt_E_XA.trt"
E_f = open(E_engine_file_path, "rb")
E_engine = runtime.deserialize_cuda_engine(E_f.read())
pD_engine_file_path = "flt_pD_XA.trt"
pD_f = open(pD_engine_file_path, "rb")
pD_engine = runtime.deserialize_cuda_engine(pD_f.read()) 

camSet = 'v4l2src device=/dev/video0 ! video/x-raw, width=352, height=288, framerate=10/1 ! videoconvert ! appsink'
cam = cv2.VideoCapture(camSet)

import socket

# Create a socket connection.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ipport = args.ip.split(':')
s.connect((ipport[0],int(ipport[1])))

lock = threading.Lock()

class VCap(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue

    def run(self):
        frame_idx = 0
        while True:
            frame_idx += 1
            _, frame = cam.read()
            # cv2.imshow('myCam', frame)
            # cv2.moveWindow('myCam', 0, 0)
            # image = frame.transpose((2,0,1))[:, 16:272, 48:304] / 255
            # input_data = np.ascontiguousarray(np.round(image[np.newaxis, :] * (2 ** 8)), dtype=np.float32)
            #y, z = EpE_exec(E_engine, input_data)
            self.data.put(frame)
            print('produce: ',frame_idx)
            while self.data.full():
                time.sleep(0.001)





frames = Queue(2)
vcap = VCap('VCap', frames)
vcap.start()

frame_idx = 0

while True:
    if not frames.empty():
        frame_idx += 1
        frame = frames.get()[16:272, 48:304, :]
        image = Image.fromarray(frame).convert("RGB")
        input_data = transforms.ToTensor()(image).unsqueeze(0).numpy()
        # input_data = np.ascontiguousarray(np.round(image[np.newaxis, :] * (2 ** 8)), dtype=np.float32)
        #y, z = EpE_exec(E_engine, input_data)
        #yint = y.astype('int8')
        
        
        y, z = EpE_exec(E_engine, input_data)
        z = torch.tensor(z.reshape(1, 128, 4, 4))
        z_strings = entropy_bottleneck.compress(z)
        z_hat = entropy_bottleneck.decompress(z_strings, z.size()[-2:]).numpy()
        sigma_hat = pD_exec(pD_engine, z_hat)
        scales_hat = torch.tensor(sigma_hat.reshape(1, 192, 16, 16))
        indexes = gaussian_conditional.build_indexes(scales_hat)
        y_strings = gaussian_conditional.compress(torch.tensor(y.reshape(1, 192, 16, 16)), indexes)
        # print(y.max())
        # sigma = pD_exec(pD_engine, z)
        print("frame %d:" % (frame_idx))
        # y_strings, z_strings = compress(y, z, sigma)
        
        jmsg = cv2.imencode("pic.jp2", frame, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 18])[1]
        #jmsg = cv2.imencode("pic.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])[1]
        bjmsg = jmsg.tobytes()
        
        jmlen = len(bjmsg)
        print(jmlen)
        msg = y_strings[0]+z_strings[0]+bytes([255, 255])+bjmsg+bytes([255, 255])
        print("len of y:", len(y_strings[0]), "len of z:", len(z_strings[0]))
        msglen = len(msg)
        msg_len_bs = struct.pack("3i", msglen, len(z_strings[0]), jmlen)
        s.send(msg_len_bs)
        print(msglen - jmlen)
        s.send(msg)
        
        # bmsg = bytes('J', 'utf-8') + bmsg
        # print(len(y_strings[0]+z_strings[0]))
        # print(len(bmsg))
        # s.send(y_strings[0]+z_strings[0])
        # s.send(bmsg)
        
        ok = s.recv(1000)
        # print(ok)
        


vcap.join()
cam.release()
cv2.destroyAllWindows()
