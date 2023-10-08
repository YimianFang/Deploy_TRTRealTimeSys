import math
from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
# from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from PIL import Image
from torchvision import transforms

# fltm = torch.load("int8model/model.pth.tar")

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    
    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return torch.max(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """
    
    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x):
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)
    
    def handle(self):
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        C = gamma.size()[0]
        gamma = gamma.reshape(C, C, 1, 1)
        self.gamma2 = nn.Parameter(gamma)
        self.beta2 = nn.Parameter(beta)

    def forward(self, x):

        norm = F.conv2d(x**2, self.gamma2, self.beta2)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class Encoder(nn.Module):
    def __init__(self, N=128, M=192):
        super(Encoder, self).__init__()
        
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        return y, z


model = Encoder().cuda()
pre = torch.load("trt_flt/EpE_dict.pth.tar")
model.load_state_dict(pre)
model.g_a[1].handle()
model.g_a[3].handle()
model.g_a[5].handle()
img = Image.open("ppm/jpgval/kodim08_patch.jpg").convert("RGB")
print(img)
input = transforms.ToTensor()(img).unsqueeze(0).cuda()
# print(input.shape)
model.eval()
outy, outz = model(input)
torch.save(outy, "trt_flt/outy.pth.tar")
torch.save(outz, "trt_flt/outz.pth.tar")

dummy_input = input.to(torch.float)
input_names = ["input"]
output_names = ["y", "z"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "trt_flt/flt_E.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)