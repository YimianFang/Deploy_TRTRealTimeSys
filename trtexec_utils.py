import os
import math
import torch
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# model_path='FashionMNIST.onnx'
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #batchsize=1
runtime = trt.Runtime(TRT_LOGGER)

y_idx = 0
z_idx = 0

# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#     builder.max_workspace_size = 1 << 28
#     builder.max_batch_size = 1
    
#     if not os.path.exists(model_path):
#         print('ONNX file {} not found.'.format(model_path))
#         exit(0)
#     print('Loading ONNX file from path {}...'.format(model_path))
#     with open(model_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         if not parser.parse(model.read()):
#             print ('ERROR: Failed to parse the ONNX file.')
#             for error in range(parser.num_errors):
#                 print (parser.get_error(error))
            
#     network.get_input(0).shape = [1, 1, 256, 256]
#     print('Completed parsing of ONNX file')
#     engine = builder.build_cuda_engine(network)
#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())
    
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()
 
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
 
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def EpE_exec(E_engine, image):
    global y_idx,z_idx
    with E_engine.create_execution_context() as E_context:
        in_idx = E_engine['input']
        y_idx = E_engine['y']
        z_idx = E_engine['z']
        E_inputs, E_outputs, E_bindings, E_stream = allocate_buffers(E_engine)
        # image = cv2.imread('123.jpg',cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image,(28,28))
        # print(image.shape)
        # image=image[np.newaxis,np.newaxis,:,:].astype(np.float32)
        # image = np.round(np.random.random([1, 3, 256, 256]) * 256).astype('float32')
        E_inputs[0].host = image
        #开始推理
        trt_outputs = do_inference_v2(E_context, bindings=E_bindings, inputs=E_inputs, outputs=E_outputs, stream=E_stream)

    # for trt_out in trt_outputs:
    #     print(trt_out.shape, trt_out)

    # y和z可直接进行编码
    y = trt_outputs[y_idx - 1]
    z = trt_outputs[z_idx - 1]
        
    return y, z

def pD_exec(pD_engine, z):
    with pD_engine.create_execution_context() as pD_context:
        pD_inputs, pD_outputs, pD_bindings, pD_stream = allocate_buffers(pD_engine)
        pD_inputs[0].host = z #.reshape(1, 128, 4, 4)
        # 开始推理
        trt_outputs = do_inference_v2(pD_context, bindings=pD_bindings, inputs=pD_inputs, outputs=pD_outputs, stream=pD_stream)
    
    # for trt_out in trt_outputs:
    #     print(trt_out.shape, trt_out)

    # 注意对sigma的处理
    sigma = np.exp(trt_outputs[0]/(2 ** 4))
    
    return sigma

def D_exec(D_engine, y):
    with D_engine.create_execution_context() as D_context:
        D_inputs, D_outputs, D_bindings, D_stream = allocate_buffers(D_engine)
        D_inputs[0].host = y #.reshape(1, 192, 16, 16)
        # 开始推理
        trt_outputs = do_inference_v2(D_context, bindings=D_bindings, inputs=D_inputs, outputs=D_outputs, stream=D_stream)
    
    # for trt_out in trt_outputs:
    #     print(trt_out.shape, trt_out)

    # Decoder的重建输出需要/255
    x_hat_255 = np.clip(trt_outputs[0], 0, 255)
    x_hat_1 = np.clip(trt_outputs[0] / 255, 0, 1)
    
    return x_hat_255, x_hat_1

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

entropy_bottleneck = EntropyBottleneck(128)
gaussian_conditional = GaussianConditional(None)

entropy_bottleneck.update()
scale_table = get_scale_table()
gaussian_conditional.update_scale_table(scale_table)

entropy_bottleneck.load_state_dict(torch.load("entropy_bottleneck.pth.tar"))
# gaussian_conditional.load_state_dict(torch.load("gaussian_conditional.pth.tar"))

def compress(y, z, sigma):
    z = torch.tensor(z.reshape(1, 128, 4, 4))
    z_strings = entropy_bottleneck.compress(z)
    z_hat = entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    # torch.save(entropy_bottleneck.state_dict(), "entropy_bottleneck.pth.tar")
    print((z - z_hat).sum())
    
    scales_hat = torch.tensor(sigma.reshape(1, 192, 16, 16))
    y = torch.tensor(y.reshape(1, 192, 16, 16))
    indexes = gaussian_conditional.build_indexes(scales_hat)
    y_strings = gaussian_conditional.compress(y, indexes)
    # torch.save(gaussian_conditional.state_dict(), "gaussian_conditional.pth.tar")
    return y_strings, z_strings

def decompress_z(z_strings, shape):
    z_hat = entropy_bottleneck.decompress(z_strings, shape)
    return z_hat

def decompress_y(sigma_hat, y_strings):
    indexes = gaussian_conditional.build_indexes(torch.tensor(sigma_hat).reshape(1, 192, 16, 16))
    y_hat = gaussian_conditional.decompress(y_strings, indexes)
    return y_hat
    
if __name__ == "__main__":
    E_engine_file_path = "int8LIC_E2.trt"
    pD_engine_file_path = "int8LIC_pD2.trt"