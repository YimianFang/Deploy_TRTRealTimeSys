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
        self.Econv1.bias.data = torch.floor(fltm["Encoder.conv1.bias"] * (2 ** 7))

        self.Econv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.Econv2.weight.data = fltm["Encoder.conv2.weight"].clamp(-127, 127).round()
        self.Econv2.bias.data = torch.floor(fltm["Encoder.conv2.bias"] / 2)

        self.Econv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.Econv3.weight.data = fltm["Encoder.conv3.weight"].clamp(-127, 127).round()
        self.Econv3.bias.data = torch.floor(fltm["Encoder.conv3.bias"] / 2)

        self.Econv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        self.Econv4.weight.data = fltm["Encoder.conv4.weight"].clamp(-127, 127).round()
        self.Econv4.bias.data = torch.floor(fltm["Encoder.conv4.bias"] / 2)

        self.pEconv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        self.pEconv1.weight.data = fltm["priorEncoder.conv1.weight"].clamp(-127, 127).round()
        self.pEconv1.bias.data = torch.floor(fltm["priorEncoder.conv1.bias"] * (2 ** 3))

        self.pEconv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.pEconv2.weight.data = fltm["priorEncoder.conv2.weight"].clamp(-127, 127).round()
        self.pEconv2.bias.data = torch.floor(fltm["priorEncoder.conv2.bias"] / 2)

        self.pEconv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.pEconv3.weight.data = fltm["priorEncoder.conv3.weight"].clamp(-127, 127).round()
        self.pEconv3.bias.data = torch.floor(fltm["priorEncoder.conv3.bias"] / 2)

        self.mulsE0 = torch.tensor(np.load("int8onnxdata/mulsE0.npy")).cuda() * 2
        self.mulsE1 = torch.tensor(np.load("int8onnxdata/mulsE1.npy")).cuda() * 2
        self.mulsE2 = torch.tensor(np.load("int8onnxdata/mulsE2.npy")).cuda() * 2
        self.mulsE3 = torch.tensor(np.load("int8onnxdata/mulsE3.npy")).cuda() * 2
        self.clpE = torch.tensor(np.load("int8onnxdata/clpE.npy")).cuda()
        self.sclE = torch.tensor(np.load("int8onnxdata/sclE.npy")).cuda()

        self.mulspE0 = torch.tensor(np.load("int8onnxdata/mulspE0.npy")).cuda() * 2
        self.mulspE1 = torch.tensor(np.load("int8onnxdata/mulspE1.npy")).cuda() * 2
        self.mulspE2 = torch.tensor(np.load("int8onnxdata/mulspE2.npy")).cuda() * 2
        self.clppE = torch.tensor(np.load("int8onnxdata/clppE.npy")).cuda()
        self.sclpE = torch.tensor(np.load("int8onnxdata/sclpE.npy")).cuda()

        # self.initial_127plus()


    def forward(self, x):
        x = self.Econv1(torch.floor(x / 2))
        x = x * self.mulsE0
        x = torch.floor((x + 524288) >> 20)
        x = torch.clip(x, 0, self.clpE[0])
        E_1 = torch.floor((x * self.sclE[0] + 262144) >> 19)

        x = self.Econv2(torch.floor(E_1 / 2))
        x = x * self.mulsE1
        x = torch.floor((x + 32768) >> 16)
        x = torch.clip(x, 0, self.clpE[1])
        E_2 = torch.floor((x * self.sclE[1] + 262144) >> 19)

        x = self.Econv3(torch.floor(E_2 / 2))
        x = x * self.mulsE2
        x = torch.floor((x + 32768) >> 16)
        x = torch.clip(x, 0, self.clpE[2])
        E_3 = torch.floor((x * self.sclE[2] + 262144) >> 19)

        x = self.Econv4(torch.floor(E_3 / 2)) ### .to(torch.int32)
        x = x * self.mulsE3
        E_4 = torch.floor((x + 16384) >> 15) #####针对ana最后一层的改进，少除2**4
        y = torch.floor((E_4 + 8) >> 4)

        x = torch.abs(E_4)
        x = torch.clip(x, 0, 255)
        x = self.pEconv1(torch.floor(x / 2)) ### .to(torch.int32)
        x = x * self.mulspE0
        x = torch.floor((x + 1024) >> 11)
        x = torch.clip(x, 0, self.clppE[0])
        pE_1 = torch.floor((x * self.sclpE[0] + 1048576) >> 21)

        x = self.pEconv2(torch.floor(pE_1 / 2)) ### .to(torch.int32)
        x = x * self.mulspE1
        x = torch.floor((x + 8192) >> 14)
        x = torch.clip(x, 0, self.clppE[1])
        pE_2 = torch.floor((x * self.sclpE[1] + 1048576) >> 21)

        x = self.pEconv3(torch.floor(pE_2 / 2)) ### .to(torch.int32)
        x = x * self.mulspE2
        z = torch.floor((x + 4194304) >> 23)

        # torch.save(E_1, "trtoutput/x86_E2_E_1.pth.tar")
        # torch.save(E_2, "trtoutput/x86_E2_E_2.pth.tar")
        # torch.save(E_3, "trtoutput/x86_E2_E_3.pth.tar")
        # torch.save(y, "trtoutput/x86_E2_y.pth.tar")
        # torch.save(z, "trtoutput/x86_E2_z.pth.tar")
        # torch.save(x, "trtoutput/x86_E2_x.pth.tar")

        return y, z, E_2, E_3, pE_1, pE_2

    def initial_127plus(self):
        self.E1_127 = torch.ones([1, 3, 256, 256], device="cuda") * 127
        self.E2_127 = torch.ones([1, 128, 128, 128], device="cuda") * 127                  
        self.E3_127 = torch.ones([1, 128, 64, 64], device="cuda") * 127
        self.E4_127 = torch.ones([1, 128, 32, 32], device="cuda") * 127
        self.pE1_127 = torch.ones([1, 192, 16, 16], device="cuda") * 127
        self.pE2_127 = torch.ones([1, 128, 16, 16], device="cuda") * 127
        self.pE3_127 = torch.ones([1, 128, 8, 8], device="cuda") * 127

model = Encoder().cuda()
input = torch.load("ppm/jpgval/kodim08_patch.pth.tar").cuda()
model.eval()
out = model(input)

dummy_input = input.to(torch.float)
input_names = ["input"]
output_names = ["y", "z"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "int8LIC_E2.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)