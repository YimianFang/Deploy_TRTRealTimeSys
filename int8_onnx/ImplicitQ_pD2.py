import math
import torch.nn as nn
import torch
from yaml import load
from quant_utils import *
import torch.nn.functional as F

fltm = torch.load("int8model/model.pth.tar")

class priorDecoder(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(priorDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv1.weight.data = torch.flip(fltm["priorDecoder.deconv1.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv1.bias.data = fltm["priorDecoder.deconv1.bias"].round()

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv2.weight.data = torch.flip(fltm["priorDecoder.deconv2.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv2.bias.data = torch.floor(fltm["priorDecoder.deconv2.bias"] / 2)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        self.deconv3.weight.data = torch.flip(fltm["priorDecoder.deconv3.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv3.bias.data = torch.floor(fltm["priorDecoder.deconv3.bias"] / 2)

        self.mulspD0 = torch.tensor(np.load("int8onnxdata/mulspD0.npy")).cuda()
        self.mulspD1 = torch.tensor(np.load("int8onnxdata/mulspD1.npy")).cuda() * 2
        self.mulspD2 = torch.tensor(np.load("int8onnxdata/mulspD2.npy")).cuda() * 2
        self.clp = torch.tensor(np.load("int8onnxdata/clppD.npy")).cuda()
        self.scl = torch.tensor(np.load("int8onnxdata/sclpD.npy")).cuda()
        # self.charts = torch.tensor(np.load("int8onnxdata/charts.npy")).cuda()

        # self.initial_127plus()

    def forward(self, x):
        x = self.deconv1(x) ### .to(torch.int32)
        x = x * self.mulspD0
        x = torch.floor((x + 2 ** 8) >> 9)
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 2 ** 20) >> 21)

        x = self.deconv2(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulspD1
        x = torch.floor((x + 2 ** 9) >> 10)
        x = torch.clip(x, 0, self.clp[1])
        x = torch.floor((x * self.scl[1] + 2 ** 20) >> 21)

        x = self.deconv3(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulspD2
        ### check charts ###
        # r1 = 14
        # r2 = 6
        x = torch.floor((x + 2 ** 13) >> 14) #int 
        #####直接输出charts第1列中的结果，通过compression/charts.npy查表进行熵编码
        #####charts.npy的形成见charts.py
        #####具体的熵编码和熵解码步骤见rangec.py
        # idx = torch.zeros_like(x).cuda().to(torch.int64)
        # for s in self.charts[:, 0]:
        #     idx += (x > s)
        # x = self.charts[idx, 1]


        # torch.save(x, "trtoutput/x86_pD2_sigma.pth.tar")

        return x

    def initial_127plus(self):
        self.D2_127plus = F.conv_transpose2d(torch.ones([1, 128, 8, 8], device="cuda") * 127,
                                    self.deconv2.weight.data,
                                    torch.zeros_like(self.deconv2.bias.data),
                                    self.deconv2.stride,
                                    self.deconv2.padding,
                                    self.deconv2.output_padding)
        self.D3_127plus = F.conv_transpose2d(torch.ones_like(self.D2_127plus, device="cuda") * 127,
                                    self.deconv3.weight.data,
                                    torch.zeros_like(self.deconv3.bias.data),
                                    self.deconv3.stride,
                                    self.deconv3.padding,
                                    self.deconv3.output_padding)


model = priorDecoder().cuda()
input = torch.load("trtoutput/x86_E2_z.pth.tar").cuda()
model.eval()
out = model(input)

dummy_input = input.to(torch.float)
input_names = ["z"]
output_names = ["sigma"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "int8LIC_pD2_charts.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)