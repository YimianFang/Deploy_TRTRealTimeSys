import math
import torch.nn as nn
import torch
from yaml import load
from quant_utils import *
import torch.nn.functional as F

fltm = torch.load("int8model/model.pth.tar")

class Encoder(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(Encoder, self).__init__()
        self.Econv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        self.Econv1.weight.data = fltm["Encoder.conv1.weight"].clamp(-127, 127).round()
        self.Econv1.bias.data = (fltm["Encoder.conv1.bias"] * (2 ** 8)).round()

        self.Econv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.Econv2.weight.data = fltm["Encoder.conv2.weight"].clamp(-127, 127).round()
        self.Econv2.bias.data = fltm["Encoder.conv2.bias"].round()

        self.Econv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.Econv3.weight.data = fltm["Encoder.conv3.weight"].clamp(-127, 127).round()
        self.Econv3.bias.data = fltm["Encoder.conv3.bias"].round()

        self.Econv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        self.Econv4.weight.data = fltm["Encoder.conv4.weight"].clamp(-127, 127).round()
        self.Econv4.bias.data = fltm["Encoder.conv4.bias"].round()

        self.pEconv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        self.pEconv1.weight.data = fltm["priorEncoder.conv1.weight"].clamp(-127, 127).round()
        self.pEconv1.bias.data = (fltm["priorEncoder.conv1.bias"] * (2 ** 4)).round()

        self.pEconv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.pEconv2.weight.data = fltm["priorEncoder.conv2.weight"].clamp(-127, 127).round()
        self.pEconv2.bias.data = fltm["priorEncoder.conv2.bias"].round()

        self.pEconv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.pEconv3.weight.data = fltm["priorEncoder.conv3.weight"].clamp(-127, 127).round()
        self.pEconv3.bias.data = fltm["priorEncoder.conv3.bias"].round()

        self.mulsE0 = torch.tensor(np.load("int8onnxdata/mulsE0.npy")).cuda()
        self.mulsE1 = torch.tensor(np.load("int8onnxdata/mulsE1.npy")).cuda()
        self.mulsE2 = torch.tensor(np.load("int8onnxdata/mulsE2.npy")).cuda()
        self.mulsE3 = torch.tensor(np.load("int8onnxdata/mulsE3.npy")).cuda()
        self.clpE = torch.tensor(np.load("int8onnxdata/clpE.npy")).cuda()
        self.sclE = torch.tensor(np.load("int8onnxdata/sclE.npy")).cuda()

        self.mulspE0 = torch.tensor(np.load("int8onnxdata/mulspE0.npy")).cuda()
        self.mulspE1 = torch.tensor(np.load("int8onnxdata/mulspE1.npy")).cuda()
        self.mulspE2 = torch.tensor(np.load("int8onnxdata/mulspE2.npy")).cuda()
        self.clppE = torch.tensor(np.load("int8onnxdata/clppE.npy")).cuda()
        self.sclpE = torch.tensor(np.load("int8onnxdata/sclpE.npy")).cuda()

        self.initial_127plus()


    def forward(self, x):
        x = self.Econv1(x - 127) ### .to(torch.int32)
        x = x + self.E1_127plus
        x = x * self.mulsE0
        x = torch.floor((x + 524288) >> 20)
        x = torch.clip(x, 0, self.clpE[0])
        E_1 = torch.floor((x * self.sclE[0] + 262144) >> 19)

        x = self.Econv2(E_1 - 127) ### .to(torch.int32)
        x = x + self.E2_127plus
        x = x * self.mulsE1
        x = torch.floor((x + 32768) >> 16)
        x = torch.clip(x, 0, self.clpE[1])
        E_2 = torch.floor((x * self.sclE[1] + 262144) >> 19)

        x = self.Econv3(E_2 - 127) ### .to(torch.int32)
        x = x + self.E3_127plus
        x = x * self.mulsE2
        x = torch.floor((x + 32768) >> 16)
        x = torch.clip(x, 0, self.clpE[2])
        E_3 = torch.floor((x * self.sclE[2] + 262144) >> 19)

        x = self.Econv4(E_3 - 127) ### .to(torch.int32)
        x = x + self.E4_127plus
        x = x * self.mulsE3
        E_4 = torch.floor((x + 16384) >> 15) #####针对ana最后一层的改进，少除2**4
        y = torch.floor((E_4 + 8) >> 4)

        x = torch.abs(E_4)
        x = torch.clip(x, 0, 254)
        x = self.pEconv1(x - 127) ### .to(torch.int32)
        x = x + self.pE1_127plus
        x = x * self.mulspE0
        x = torch.floor((x + 1024) >> 11)
        x = torch.clip(x, 0, self.clppE[0])
        pE_1 = torch.floor((x * self.sclpE[0] + 1048576) >> 21)

        x = self.pEconv2(pE_1 - 127) ### .to(torch.int32)
        x = x + self.pE2_127plus
        x = x * self.mulspE1
        x = torch.floor((x + 8192) >> 14)
        x = torch.clip(x, 0, self.clppE[1])
        pE_2 = torch.floor((x * self.sclpE[1] + 1048576) >> 21)

        x = self.pEconv3(pE_2 - 127) ### .to(torch.int32)
        x = x + self.pE3_127plus
        x = x * self.mulspE2
        z = torch.floor((x + 4194304) >> 23)

        # torch.save(E_1, "trtoutput/x86_E_E_1.pth.tar")
        # torch.save(E_2, "trtoutput/x86_E_E_2.pth.tar")
        # torch.save(E_3, "trtoutput/x86_E_E_3.pth.tar")
        # torch.save(y, "trtoutput/x86_E_y.pth.tar")
        # torch.save(z, "trtoutput/x86_E_z.pth.tar")

        return E_2, E_3, y, pE_1, pE_2, z

    def initial_127plus(self):
        self.E1_127plus = F.conv2d(torch.ones([1, 3, 256, 256], device="cuda") * 127,
                                    self.Econv1.weight.data,
                                    torch.zeros_like(self.Econv1.bias.data),
                                    self.Econv1.stride,
                                    self.Econv1.padding)
        self.E2_127plus = F.conv2d(torch.ones_like(self.E1_127plus, device="cuda") * 127,
                                    self.Econv2.weight.data,
                                    torch.zeros_like(self.Econv2.bias.data),
                                    self.Econv2.stride,
                                    self.Econv2.padding)
        self.E3_127plus = F.conv2d(torch.ones_like(self.E2_127plus, device="cuda") * 127,
                                    self.Econv3.weight.data,
                                    torch.zeros_like(self.Econv3.bias.data),
                                    self.Econv3.stride,
                                    self.Econv3.padding)
        self.E4_127plus = F.conv2d(torch.ones_like(self.E3_127plus, device="cuda") * 127,
                                    self.Econv4.weight.data,
                                    torch.zeros_like(self.Econv4.bias.data),
                                    self.Econv4.stride,
                                    self.Econv4.padding)
        self.pE1_127plus = F.conv2d(torch.ones_like(self.E4_127plus, device="cuda") * 127,
                                    self.pEconv1.weight.data,
                                    torch.zeros_like(self.pEconv1.bias.data),
                                    self.pEconv1.stride,
                                    self.pEconv1.padding)
        self.pE2_127plus = F.conv2d(torch.ones_like(self.pE1_127plus, device="cuda") * 127,
                                    self.pEconv2.weight.data,
                                    torch.zeros_like(self.pEconv2.bias.data),
                                    self.pEconv2.stride,
                                    self.pEconv2.padding)
        self.pE3_127plus = F.conv2d(torch.ones_like(self.pE2_127plus, device="cuda") * 127,
                                    self.pEconv3.weight.data,
                                    torch.zeros_like(self.pEconv3.bias.data),
                                    self.pEconv3.stride,
                                    self.pEconv3.padding)

model = Encoder().cuda()
input = torch.load("ppm/jpgval/kodim08_patch.pth.tar").cuda()
model.eval()
out = model(input)

dummy_input = input.to(torch.float)
input_names = ["input"]
output_names = ["E_2", "E_3", "y", "pE_1", "pE_2", "z"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "int8LIC_E.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)