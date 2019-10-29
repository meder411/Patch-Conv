import torch
import torch.nn as nn
from patch_convolution import *
from torch.autograd import gradcheck

B = 1
P = 4
H = 2
W = 2

in_channels = 2
out_channels = 3
kernel_size = 3
padding = 1
stride = 1
dilation = 1
cuda = False

# Initialize networks with same parameters
torch_conv = nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding=padding,
                       stride=stride,
                       dilation=dilation).double()
torch_conv.weight.data = torch.ones_like(torch_conv.weight)
torch_conv.bias.data = torch.zeros_like(torch_conv.bias)
patch_conv = PatchConvolution(in_channels=in_channels,
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
    print('  Same output:', bool((patch_out[:, :, i, ...] == out).all()))

gradcheck_res = gradcheck(patch_conv, x)
print('Passed gradcheck:', gradcheck_res)