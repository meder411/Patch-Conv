from patch_convolution import *
import torch
import torch.nn as nn
import time

# ---------------
# Parameters
# ---------------
# Number of profile iterations to run
itt = 20

# Input and conv parameters
B = 1
P = 20480
H = 4
W = 4
in_channels = 64
out_channels = 128
kernel_size = 3
padding = 1
stride = 1
dilation = 1
cuda = True

# --------------------
# Layers to compare
# --------------------
# Patch convolution using torch.nn.unfold + torch.matmul
pmm = PatchMMConvolution(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         stride=stride,
                         dilation=dilation).double()

# Standard torch.nn.Conv2D but with patches rolled into batch dimension
# e.g. B x P x C x H x W --> BP x C x H x W
conv = nn.Conv2d(in_channels=in_channels,
                 out_channels=out_channels,
                 kernel_size=kernel_size,
                 padding=padding,
                 stride=stride,
                 dilation=dilation).double()

# Convolution with my CUDA implementation of patch_im2col / patch_col2im + cuBLAS GEMM operations
patch_conv = PatchConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              dilation=dilation).double()

# Set weights and biases to be the same
weight = torch.ones_like(pmm.weight)
bias = torch.ones_like(pmm.bias) * 2
pmm.weight.data = weight
pmm.bias.data = bias
conv.weight.data = weight
conv.bias.data = bias
patch_conv.weight.data = weight
patch_conv.bias.data = bias

# ---------------------
# Create inputs
# ---------------------
# Batches of patches input (B x P x C x H x W)
x = torch.arange(1 * P * 1 * H * W).view(1, P, 1, H,
                                         W).repeat(B, 1, in_channels, 1,
                                                   1).double()
# Standard input format (B x C x H x W) with a similar (actually greater) number of total elements
x_sim = torch.rand(B, in_channels, 600, 600).double()

# Put on GPU if desired
if cuda:
    pmm = pmm.cuda()
    patch_conv = patch_conv.cuda()
    conv = conv.cuda()
    x = x.cuda()
    x_sim = x_sim.cuda()


def profile_layer(x, layer, itt=10):
    """
    Runs a forward pass on the layer <itt> times and returns the average time
    """
    time_accum = 0.0
    for i in range(itt):
        # Time the forward execution
        if cuda:
            torch.cuda.synchronize()
        s = time.time()
        out = layer(x)
        if cuda:
            torch.cuda.synchronize()
        e = time.time()
        time_accum += (e - s)
    return time_accum / itt


# -----------------------
# Profile the layers
# -----------------------
print('\n')
print('Various patch convolution methods:')
print('  Using-unfold avg. time: {:.6f} seconds'.format(
    profile_layer(x, pmm, itt)))
print('  Rolled-into-batch avg. time: {:.6f} seconds'.format(
    profile_layer(x.view(-1, in_channels, H, W), conv, itt)))
print('  Custom-patch-im2col avg. time: {:.6f} seconds'.format(
    profile_layer(x.transpose(1, 2).contiguous(), patch_conv, itt)))

print('\n')
print(
    'Compare to traditional convolution with B x C x H x W inputs with similar number (actually more) of elements:'
)
print('  nn.Conv2d: {:.6f} seconds'.format(profile_layer(x_sim, conv)))

# ----------------
# Sanity check
# ----------------
# Compare outputs
pmm_out = pmm(x)
conv_out = conv(x.view(-1, in_channels, H, W))
conv_out = conv_out.view(B, P, out_channels, *conv_out.shape[-2:])
patch_conv_out = patch_conv(x.transpose(1, 2).contiguous())
patch_conv_out = patch_conv_out.transpose(1, 2).contiguous()
print('\nSanity Check: All Equivalent = ',
      bool(((pmm_out - conv_out) < 1e-8).all()
           and ((patch_conv_out - conv_out) < 1e-8).all()))