import cv2
import numpy as np
from trtexec_utils import *
from yz_trans.alts_server import build_svrc

def psnr(img1, img2):
   mse = np.mean((img1 - img2) ** 2)
   return 10 * math.log10(1.0 ** 2 / mse)

pD_engine_file_path = "int8LIC_pD2.trt"
pD_f = open(pD_engine_file_path, "rb")
pD_engine = runtime.deserialize_cuda_engine(pD_f.read()) 
D_engine_file_path = "int8LIC_D2.trt"
D_f = open(D_engine_file_path, "rb")
D_engine = runtime.deserialize_cuda_engine(D_f.read()) 

svr, svrc = build_svrc("0.0.0.0:50051")

frame_idx = 0

fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('recon_x.mp4', fourcc, 30.0, (256, 256),True)

while True:
    if len(svrc.q.queue) > 0:
        frame_idx += 1
        merge_trans = svrc.q.queue.pop()
        y_trans, z_trans = merge_trans.split("//")
        y_strings = [bytes(map(int, y_trans.split(' ')))]
        z_strings = [bytes(map(int, z_trans.split(' ')))]
        # print(z_strings)
        
        z_hat = np.array(decompress_z(z_strings, (4, 4)))
        sigma_hat = pD_exec(pD_engine, z_hat)
        
        y_hat = np.array(decompress_y(sigma_hat, y_strings))
        print("frame %d:" % (frame_idx), y_hat.max(), z_hat.max(), sigma_hat.max())
        x_hat_255, x_hat_1 = D_exec(D_engine, y_hat)
        
        recon_x = np.transpose(x_hat_255.reshape(3, 256, 256), (1, 2, 0)).astype(np.uint8)
        # cv2.imwrite("recon_x.jpg", recon_x)
        out.write(recon_x)
    else:
        svr.wait_for_termination(timeout=10)
        if len(svrc.q.queue) == 0:
            print("------------------Time Out - Stop Decoder------------------")
            break

out.release()
cv2.destroyAllWindows()