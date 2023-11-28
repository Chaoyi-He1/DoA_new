from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath
from .positional_embedding import build_position_encoding


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "leaky_relu":
        return F.leaky_relu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def calculate_conv1d_padding(stride, kernel_size, d_in, d_out, dilation=1):
    """
    Calculate the padding value for a 1D convolutional layer.

    Args:
        stride (int): Stride value for the convolutional layer.
        kernel_size (int): Kernel size for the convolutional layer.
        d_in (int): Input dimension of the feature map.
        d_out (int): Output dimension of the feature map.
        dilation (int, optional): Dilation value for the convolutional layer.
                                  Default is 1.

    Returns:
        int: Padding value for the convolutional layer.

    """
    padding = math.ceil((stride * (d_out - 1) - 
                         d_in + (dilation * 
                                 (kernel_size - 1)) + 1) / 2)
    assert padding >= 0, "Padding value must be greater than or equal to 0."

    return padding


def calculate_conv2d_padding(stride, kernel_size, d_in, d_out, dilation=1):
    """
    Calculate the padding value for a 2D convolutional layer.
    
    Arguments:
    - stride (int or tuple): The stride value(s) for the convolution.
    - kernel_size (int or tuple): The size of the convolutional kernel.
    - d_in (tuple): The input dimensions (height, width) of the feature map.
    - d_out (tuple): The output dimensions (height, width) of the feature map.
    - dilation (int or tuple): The dilation value(s) for the convolution. Default is 1.
    
    Returns:
    - padding (tuple): The padding value(s) (padding_h, padding_w) for the convolution.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in, w_in = d_in
    h_out, w_out = d_out
    h_k, w_k = kernel_size
    h_s, w_s = stride
    h_d, w_d = dilation

    padding_h = math.ceil(((h_out - 1) * h_s + h_k - h_in + (h_k - 1) * (h_d - 1)) / 2)
    padding_w = math.ceil(((w_out - 1) * w_s + w_k - w_in + (w_k - 1) * (w_d - 1)) / 2)
    assert padding_h >= 0 and padding_w >= 0, "Padding value(s) cannot be negative."

    padding = (padding_h, padding_w)
    return padding


class Conv1d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv1d_BN_Relu, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv1d_BN_Relu, self).forward(x)


class Conv2d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d_BN_Relu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv2d_BN_Relu, self).forward(x)


