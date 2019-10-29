import torch
import torch.nn as nn

import math
from .layer_utils import _pair


class PatchMMConvolution(nn.Module):

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Filter size
            stride=1,  # Stride
            padding=0,  # Padding
            dilation=1):  # Dilation

        super(PatchMMConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        # Initialize parameters of the layer
        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding,
                                self.stride)
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels,
                         self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        x : B x P x C x H x W
        """
        B, P, C, IH, IW = x.shape

        # Compute output dimensions
        OH = ((IH + 2 * self.padding[0] - self.kernel_size[0] -
               (self.kernel_size[0] - 1) *
               (self.dilation[0] - 1)) // self.stride[0]) + 1
        OW = ((IW + 2 * self.padding[1] - self.kernel_size[1] -
               (self.kernel_size[1] - 1) *
               (self.dilation[1] - 1)) // self.stride[1]) + 1

        # Unfold into a (B x PCK^2 x conv_patch_size) matrix (basically col matrix) and convolve via matrix multiplication
        col = self.unfold(x.view(x.shape[0], -1, *x.shape[-2:]))
        out = torch.matmul(self.weight.view(self.out_channels, -1),
                           col.view(B, P, -1, col.shape[-1]))
        out = out.view(B, P, self.out_channels, OH, OW)

        # Add the bias
        out += self.bias.view(1, 1, -1, 1, 1)

        return out