# TRT_Deploy_RealTimeSys

## Environment Requirements
1. python 3.6.9
2. pycuda 2022.1
3. tensorrt 7.1.3.4
4. opencv 4.1.1
5. grpc 1.48.2
6. compressai 1.2.4
7. torch 1.9.0
8. numpy 1.19.5

## Tips
1. make sure __ip address__ works.
2. make sure __`.trt` files__ match the platform. If not, generate corresponding `.trt` files from `.onnx` models with `build_D_trt.py` and `build_pD_trt.py`.
3. The input size is fixed, which is `[1, 3, 256, 256]`.

## Steps to run
### Run server first
```bash
python3 main_decoder.py --ip 192.168.1.188:50051
```

### Run client then on another machine
```bash
python3 main_encoder.py --ip 192.168.1.188:50051
```
* The ip addresses on two machines should keep the same.

### Stop the program
Press the key `q` on the keyboard.