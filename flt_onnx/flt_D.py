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

class Decoder(nn.Module):
    def __init__(self, N=128, M=192):
        super(Decoder, self).__init__()
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        x = self.g_s(x)
        return x


model = Decoder().cuda()
pre = torch.load("trt_flt/D_dict.pth.tar")
model.load_state_dict(pre)
model.g_s[1].handle()
model.g_s[3].handle()
model.g_s[5].handle()
input = torch.load("trt_flt/outy.pth.tar").cuda().int().float()
model.eval()
out = model(input)
print(out.shape)
xhat = transforms.ToPILImage()(out.squeeze(0))
xhat.save("trt_flt/xhat.jpg")

dummy_input = input.to(torch.float)
input_names = ["y"]
output_names = ["x"]

# model.eval()
# output = model(input_image)
torch.onnx.export(model, dummy_input, "trt_flt/flt_D.onnx", verbose=True, export_params=True, input_names=input_names, output_names=output_names, opset_version=9) #####, dynamic_axes=dynamic_axes)