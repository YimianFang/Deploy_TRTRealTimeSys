# Deploy_TRTRealTimeSys

## Purpose
Capture a video from the camera on a platform (etc. Jetson Xavier) and the compressed and encoded frames are transmitted to another paltform (etc. Jetson TX), where the frames are decoded and reconstructed in real time.

## Environment Requirements
1. python 3.6.9
2. pycuda 2022.1
3. tensorrt 7.1.3.4
4. opencv 4.1.1
5. grpc 1.48.2
6. compressai 1.2.4
7. torch 1.9.0
8. torchvision 1.10.0
9. numpy 1.19.5

## Tips
1. make sure the __ip address__ works.
2. make sure __`.trt` files__ match the platform.
3. Folders `int8/flt_onnx` generate `.onnx` models with executing `ImplicitQ/flt_*.py` directly.
4. Folders `Int8LIC_*2/127` build **Int8** mode TensorRT engines (`.trt` files) from `.onnx` models, where `2` means `/2` and folder `127` means `-127`; `build_*_trt.py` builds **Float32** mode TensorRT engines from `.onnx` models.
5. The input size is fixed, which is `[1, 3, 256, 256]`.
6. TX2 platform **can not** support Int8 mode of TensorRT.
7. `flt_decoder_cmp.py` & `flt_encoder_cmp.py`: floating-point pretrained models with `flt_*.onnx` & `flt_*_TX/XA.trt`.
8. `decoder_cmp.py` & `encoder_cmp.py`: int8 quantized models with `Int8LIC_*.onnx` & `Int8LIC_*_TX/XA.trt`.
9. `flt_decoder_cmp.py` & `decoder_cmp.py` use `D` & `pD`, while `flt_encoder_cmp.py` & `encoder_cmp.py` use `E` & `pD`.

## Steps to run
### Run server first
```bash
python3 flt_decoder_cmp.py --ip 192.168.1.188:50051
```

### Run client then on another machine
```bash
python3 flt_encoder_cmp.py --ip 192.168.1.188:50051
```
* The ip addresses on two machines should keep the same.
