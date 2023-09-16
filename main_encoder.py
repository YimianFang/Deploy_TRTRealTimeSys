import cv2
import numpy as np
from trtexec_utils import *
from yz_trans.alts_client import build_stub, yz_trans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default="192.168.1.188:50051")
args = parser.parse_args()

E_engine_file_path = "int8LIC_E2_XA.trt"
E_f = open(E_engine_file_path, "rb")
E_engine = runtime.deserialize_cuda_engine(E_f.read())
pD_engine_file_path = "int8LIC_pD2_XA.trt"
pD_f = open(pD_engine_file_path, "rb")
pD_engine = runtime.deserialize_cuda_engine(pD_f.read()) 

camSet = 'v4l2src device=/dev/video0 ! video/x-raw, width=352, height=288, framerate=30/1 ! videoconvert ! appsink'
cam = cv2.VideoCapture(camSet)

stub = build_stub(args.ip)

frame_idx = 0

while True:
    frame_idx += 1
    _, frame = cam.read()
    # cv2.imshow('myCam', frame)
    # cv2.moveWindow('myCam', 0, 0)
    image = frame[:,:,::-1].transpose((2,0,1))[:, 16:272, 48:304] / 255
    input_data = np.ascontiguousarray(np.round(image[np.newaxis, :] * (2 ** 8)), dtype=np.float32)
    
    ####### test reconstructed image #######
    # input_data = np.round(np.random.random([1, 3, 256, 256]) * 256).astype('float32')
    # input_data = np.ones(3*256*256, dtype=np.float32)
    # f = open("kodim08_patch.txt")
    # for i in range(3*256*256):
    #     input_data[i] = float(f.readline())
    # np.save("input_data.npy", input_data)
    # input_data = np.load("input_data.npy")
    ####### test reconstructed image #######
    
    print(input_data.sum())
    
    y, z = EpE_exec(E_engine, input_data)
    sigma = pD_exec(pD_engine, z)
    print("frame %d:" % (frame_idx), y.max(), z.max(), sigma.max())
    
    y_strings, z_strings = compress(y, z, sigma)
    
    y_trans = ' '.join(map(str, (map(int, y_strings[0]))))
    # bytes(map(int, y_trans.split(' ')))
    z_trans = ' '.join(map(str, (map(int, z_strings[0]))))
    # print(z_trans)
    # print(z_strings)
    # print(type(z_strings))
    merge_trans = "//".join([y_trans, z_trans])
    
    yz_trans(stub, merge_trans)
    
    ####### debug #######
    # break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()