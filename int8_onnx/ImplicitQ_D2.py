import math
import torch.nn as nn
import torch
from yaml import load
from quant_utils import *
import torch.nn.functional as F

fltm = torch.load("int8model/model.pth.tar")

class Decoder(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv1.weight.data = torch.flip(fltm["Decoder.deconv1.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv1.bias.data = fltm["Decoder.deconv1.bias"].round()

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv2.weight.data = torch.flip(fltm["Decoder.deconv2.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv2.bias.data = torch.floor(fltm["Decoder.deconv2.bias"] / 2)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv3.weight.data = torch.flip(fltm["Decoder.deconv3.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv3.bias.data = torch.floor(fltm["Decoder.deconv3.bias"] / 2)

        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        self.deconv4.weight.data = torch.flip(fltm["Decoder.deconv4.weight"].clamp(-127, 127).round().permute(1, 0, 2, 3), [2,3])
        self.deconv4.bias.data = torch.floor(fltm["Decoder.deconv4.bias"] / 2)

        self.mulsD0 = torch.tensor(np.load("int8onnxdata/mulsD0.npy")).cuda()
        self.mulsD1 = torch.tensor(np.load("int8onnxdata/mulsD1.npy")).cuda() * 2
        self.mulsD2 = torch.tensor(np.load("int8onnxdata/mulsD2.npy")).cuda() * 2
        self.mulsD3 = torch.tensor(np.load("int8onnxdata/mulsD3.npy")).cuda() * 2
        self.clp = torch.tensor(np.load("int8onnxdata/clpD.npy")).cuda()
        self.scl = torch.tensor(np.load("int8onnxdata/sclD.npy")).cuda()

        # self.initial_127plus()

    def forward(self, x):
        x = self.deconv1(x) ### .to(torch.int32)
        x = x * self.mulsD0
        x = torch.floor((x + 4096) >> 13)
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 524288) >> 20) ###.clamp(0, 254)

        x = self.deconv2(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulsD1
        x = torch.floor((x + 4096) >> 13)
        x = torch.clip(x, 0, self.clp[1])
        x = torch.floor((x * self.scl[1] + 131072) >> 18) ###.clamp(0, 254)

        x = self.deconv3(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulsD2
        x = torch.floor((x + 4096) >> 13)
        x = torch.clip(x, 0, self.clp[2])
        x = torch.floor((x * self.scl[2] + 131072) >> 18) ###.clamp(0, 254)

        x = self.deconv4(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulsD3
        x = torch.floor((x + 2 ** 21) >> 22) #####/255 未除以255

        # torch.save(x, "trtoutput/x86_D2_xhat.pth.tar")

        return x

    def initial_127plus(self):
        self.D2_127plus = F.conv_transpose2d(torch.ones([1, 128, 32, 32], device="cuda") * 127,
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
        self.D4_127plus = F.conv_transpose2d(torch.ones_like(self.D3_127plus, device="cuda") * 127,
                                    self.deconv4.weight.data,
                                    torch.zeros_like(self.deconv4.bias.data),
                                    self.deconv4.stride,
                                    self.deconv4.padding,
                                    self.deconv4.output_padding)


model = Decoder().cuda()
input = torch.load("trtoutput/x86_E2_y.pth.tar").cuda()
model.eval()
out = model(input)

dummy_input = input.to(torch.float)
input_names = ["y"]
output_names = ["x"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "int8LIC_D2.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)