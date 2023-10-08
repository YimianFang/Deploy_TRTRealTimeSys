import torch
m = torch.load("trt_flt/fltmodel.pth.tar")

entropy_bottleneck_dict = {}
gaussian_conditional_dict = {}
EpE_dict = {}
D_dict = {}
pD_dict = {}

for k, v in m.items():
    if "entropy_bottleneck" in k:
        entropy_bottleneck_dict[k.split(".", 1)[1]] = v
    elif "gaussian_conditional" in k:
        gaussian_conditional_dict[k.split(".", 1)[1]] = v
    elif "g_s" in k:
        D_dict[k] = v
    elif "h_s" in k:
        pD_dict[k] = v
    else:
        EpE_dict[k] = v

print(len(entropy_bottleneck_dict) + len(gaussian_conditional_dict) + len(D_dict) + len(pD_dict) + len(EpE_dict))
print(len(m))

torch.save(entropy_bottleneck_dict, "trt_flt/flt_entropy_bottleneck.pth.tar")
torch.save(gaussian_conditional_dict, "trt_flt/flt_gaussian_conditional.pth.tar")
torch.save(D_dict, "trt_flt/D_dict.pth.tar")
torch.save(pD_dict, "trt_flt/pD_dict.pth.tar")
torch.save(EpE_dict, "trt_flt/EpE_dict.pth.tar")