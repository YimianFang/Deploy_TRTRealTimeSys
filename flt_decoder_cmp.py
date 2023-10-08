import os
os.environ['OPENCV_IO_ENABLE_JASPER']= 'True'
import cv2
import numpy as np
from flt_trtexec_utils import *
#from yz_trans.alts_server import build_svrc
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default="192.168.1.188:50051")
args = parser.parse_args()

import pickle
import socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ipport = args.ip.split(':')
print(ipport)
server.bind((ipport[0],int(ipport[1])))
server.listen(1)
conn, addr = server.accept()
print ('Connected by ', addr)


from multiprocessing import Process, Queue

import threading, struct
lock = threading.Lock()

def producerun(queue, queueJ, queuelen):
    frame_idx = 0
    while conn:
        frame_idx += 1
        bs_msg_len = conn.recv(12)
        bs_len, zlen, blen = struct.unpack("3i", bs_msg_len)
        print(bs_len)
        s_byte = bytes()
        while len(s_byte) < bs_len:
          s_byte = s_byte + conn.recv(bs_len)
        conn.send(bytes([64, 64]))
        print("len of s_byte", len(s_byte))
        
        print(s_byte[-2:])
        j_byte = s_byte[-blen-2:-2]
        print(s_byte[-blen-4:-blen-2])
        s_byte = s_byte[:-blen-4]
        tn = len(s_byte)
        y_s, z_s = s_byte[:tn-zlen] , s_byte[tn-zlen:]
        print("len of y:", len(y_s))
        print("len of z:", len(z_s))
        queue.put((y_s,z_s))
        imgJ = cv2.imdecode(np.frombuffer(j_byte, np.uint8), cv2.IMREAD_COLOR)
        queueJ.put(imgJ)
        queuelen.put((bs_len - blen, blen))
        
        # tn = len(s_byte)
        # y_s,z_s = s_byte[:tn-1396] , s_byte[tn-1396:]
        # queue.put((y_s,z_s))
        # s_byte = conn.recv(999999)
        # conn.send(bytes([64,64]))
        # j_byte = s_byte[1:]
        # print("len of j_byte", len(s_byte))
        # imgJ = cv2.imdecode(np.frombuffer(j_byte, np.uint8), cv2.IMREAD_COLOR)
        # queueJ.put(imgJ)
        
        #print('produce: ',frame_idx)
        while queue.full():
            time.sleep(0.001)




def psnr(img1, img2):
   mse = np.mean((img1 - img2) ** 2)
   return 10 * math.log10(1.0 ** 2 / mse)

pD_engine_file_path = "flt_pD_TX.trt"
pD_f = open(pD_engine_file_path, "rb")
pD_engine = runtime.deserialize_cuda_engine(pD_f.read()) 
D_engine_file_path = "flt_D_TX.trt"
D_f = open(D_engine_file_path, "rb")
D_engine = runtime.deserialize_cuda_engine(D_f.read()) 


def decomyhat(queue):
    frame_idx = 0
    while True:
      if not queue.empty():
        y_hat = np.array(decompress_y(sigma_hat, [y_strings]))
        while queue.full():
            time.sleep(0.001)

def comsumerun(queue, queueJ, queuelen):
    frame_idx = 0
    T_start = time.time()
    ref_F = frame_idx
    while True:
      if not queue.empty():
        if frame_idx%20 == 1: 
            T_start = time.time()
            ref_F = frame_idx
        frame_idx += 1
    
        y_strings,z_strings = queue.get()
    
        z_hat = np.array(decompress_z([z_strings], (4, 4)))
        sigma_hat = pD_exec(pD_engine, z_hat) #0.02
        
        y_hat = np.array(decompress_y(sigma_hat, [y_strings])) #0.05
        x_hat = D_exec(D_engine, y_hat) #0.08
        
        # recon_x = np.transpose((x_hat * 255).reshape(3, 256, 256), (1, 2, 0)).astype(np.uint8)
        recon_x = np.transpose(x_hat.reshape(3, 256, 256), (1, 2, 0))
        
        imgJ = queueJ.get()
        
        yzlen, jlen = queuelen.get()
        
        # cv2.imwrite("recon_x.jpg", recon_x)
        # out.write(recon_x)
        
        resize_recon_x = cv2.resize(recon_x, (512,512), interpolation=4)
        resize_imgJ = cv2.resize(imgJ, (512,512), interpolation=3)
        Frate = (frame_idx-ref_F)/(time.time() - T_start)
        text = 'frame rate: {:.3f}'.format(Frate)
        org = (10, 20)
        fontFace = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 0.5
        fontcolor = (255, 0, 255) # BGRS
        thickness = 1 
        lineType = 4
        bottomLeftOrigin = 1
        #cv2.putText(resize_recon_x, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
        cv2.putText(resize_recon_x, 'bytes: {:d}'.format(yzlen), org, fontFace, fontScale, fontcolor, thickness, lineType)
        cv2.putText(resize_imgJ, 'bytes: {:d}'.format(jlen), org, fontFace, fontScale, fontcolor, thickness, lineType)
        cv2.imshow("recon_x", resize_recon_x)
        cv2.imshow("JPEG2000", resize_imgJ)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        print("frame %d:" % (frame_idx),text)    


frames = Queue(1)
framesJ = Queue(1)
frameslen = Queue(1)
#yhats = Queue(1)
vcap = Process(target=producerun, args=(frames, framesJ, frameslen))
vcap.start()
#show = Process(target=comsumerun, args=(frames,))
#show.start()
try:
  comsumerun(frames, framesJ, frameslen)
except Exception as e:
  print(e)
  pass
vcap.join()
#show.join()
cv2.destroyAllWindows()
conn.close()
