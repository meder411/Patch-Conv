import torch
import torch.nn as nn
from patch_convolution import *
from torch.autograd import gradcheck

B = 2
P = 4
H = 5
W = 5

in_channels = 3
out_channels = 4
kernel_size = 4
padding = 1
stride = 2
dilation = 1
cuda = False

# Initialize networks with same parameters
torch_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                dilation=dilation).double()
torch_conv.weight.data = torch.ones_like(torch_conv.weight)
torch_conv.bias.data = torch.zeros_like(torch_conv.bias)
patch_conv = TransposedPatchConvolution(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        dilation=dilation).double()
patch_conv.weight.data = torch.ones_like(patch_conv.weight)
patch_conv.bias.data = torch.zeros_like(patch_conv.bias)

# Input
x = torch.arange(in_channels * H * W).view(1, in_channels, 1, H,
                                           W).repeat(B, 1, P, 1, 1).double()
x.requires_grad = True

# Put on GPU if desired
if cuda:
    x = x.cuda()
    patch_conv = patch_conv.cuda()
    torch_conv = torch_conv.cuda()

# Run my patch convolution
patch_out = patch_conv(x)

# Compare my patch convolution to the traditional convolution on each patch (for equivalence)
print('Checking each patch...')
for i in range(P):
    out = torch_conv(x[:, :, i, ...])
    print('  Same output: ', bool(
        (patch_out[:, :, i, ...] == out).all().item()))

gradcheck_res = gradcheck(patch_conv, x)
print('Passed gradcheck:', gradcheck_res)