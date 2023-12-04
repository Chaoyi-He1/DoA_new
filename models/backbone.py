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


class ResBlock_1d(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: int = 1024, dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_1d, self).__init__()
        pad = calculate_conv1d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv1d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv1d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.conv2(self.conv1(x)))
    

class ResBlock_2d(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: Tuple[int, int] = (256, 1024), dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_2d, self).__init__()
        pad = calculate_conv2d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv2d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv2d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.conv2(self.conv1(x)))


class ResBlock_1d_with_Attention(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: int = 1024, dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_1d_with_Attention, self).__init__()
        pad = calculate_conv1d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv1d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv1d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.atten = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv2(self.conv1(x))
        atten_out = F.sigmoid(self.atten(x))
        return x * atten_out + self.drop_path(conv_out)


class ResBlock_2d_with_Attention(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: Tuple[int, int] = (256, 1024), dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_2d_with_Attention, self).__init__()
        pad = calculate_conv2d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv2d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv2d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.atten = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv2(self.conv1(x))
        atten_out = F.sigmoid(self.atten(x))
        return x * atten_out + self.drop_path(conv_out)


class Conv1d_AutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 1024, in_channel: int = 12, 
                 drop_path: float = 0.4, with_atten: bool = True) -> None:
        super(Conv1d_AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.temp_dim = in_dim
        self.channel = 16
        pad = calculate_conv1d_padding(1, 11, self.temp_dim, self.temp_dim)
        self.conv1 = Conv1d_BN_Relu(self.in_channel, self.channel, kernel_size=11, padding=pad)
        pad = calculate_conv1d_padding(2, 13, self.temp_dim, self.temp_dim // 2)
        self.conv2 = Conv1d_BN_Relu(self.channel, self.channel * 2, kernel_size=13, stride=2, padding=pad)
        self.channel *= 2
        self.temp_dim //= 2

        self.ResNet = nn.ModuleList()
        res_params = list(zip([2, 2, 4, 4, 2], [7, 7, 9, 9, 11],   # num_blocks, kernel_size
                              [3, 3, 3, 3, 3], [1, 5, 5, 3, 3]))   # stride, dilation
        # final channels = 512; final temp_dim = in_dim // (2^5) = in_dim // 32
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_1d(self.channel, kernel_size, stride, self.temp_dim, dilation,
                                            drop_path) if not with_atten else \
                                ResBlock_1d_with_Attention(self.channel, kernel_size, stride, self.temp_dim,
                                                           dilation, drop_path)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv1d_padding(stride, kernel_size, self.temp_dim, self.temp_dim // 2, dilation)
                self.ResNet.append(Conv1d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim //= 2
        
        self.ResNet.append(nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=1))
        
        self.reduce_temp_dim = nn.Sequential(
            nn.Linear(self.temp_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, inputs: Tensor) -> Tensor:
        '''
        inputs: [L, channel, in_dim],
                L is the number of frames in each input data matrix, default is 512
                channel is the number of channels in each input data matrix, 
                    default is 12, which means 4 receivers antenna's T-F data (Amp, Real, Imag)
                in_dim is the number of frequency bins in each input data matrix, default is 512
        output: [L, Embedding], Embedding = 512
        '''

        x = self.conv1(inputs)
        x = self.conv2(x)
        for block in self.ResNet:
            x = block(x)
        x = self.reduce_temp_dim(x)
        x = x.squeeze()
        return x


class Conv2d_AutoEncoder(nn.Module):
    def __init__(self, in_dim: Tuple[int, int] = (256, 1024), in_channel: int = 12, 
                 drop_path: float = 0.4, with_atten: bool = True) -> None:
        super(Conv2d_AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.temp_dim = in_dim
        self.channel = 16
        pad = calculate_conv2d_padding(1, 11, self.temp_dim, self.temp_dim)
        self.conv1 = Conv2d_BN_Relu(self.in_channel, self.channel, kernel_size=11, padding=pad)
        pad = calculate_conv2d_padding(2, 13, self.temp_dim, tuple(element // 2 for element in self.temp_dim))
        self.conv2 = Conv2d_BN_Relu(self.channel, self.channel * 2, kernel_size=13, stride=2, padding=pad)
        self.channel *= 2
        self.temp_dim = tuple(element // 2 for element in self.temp_dim)

        self.ResNet = nn.ModuleList()
        res_params = list(zip([4, 6, 8, 8, 4], [3, 7, 9, 9, 11],   # num_blocks, kernel_size
                              [3, 3, 3, 3, 3], [1, 5, 5, 3, 3]))   # stride, dilation
        # final channels = 512; final temp_dim = in_dim // (2^5) = in_dim // 32
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_2d(self.channel, kernel_size, stride, self.temp_dim, dilation,
                                            drop_path) if not with_atten else \
                                ResBlock_2d_with_Attention(self.channel, kernel_size, stride, self.temp_dim,
                                                           dilation, drop_path)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv2d_padding(stride, kernel_size, self.temp_dim, 
                                               tuple(element // 2 for element in self.temp_dim), dilation)
                self.ResNet.append(Conv2d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, stride, pad, dilation))
                self.channel *= 2
                self.temp_dim = tuple(element // 2 for element in self.temp_dim)

        self.flat = nn.Flatten(start_dim=-2, end_dim=-1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        inputs: [B, channel, L, in_dim],
                B is the batch size
                L is the number of frames in each input data matrix, default is 512
                channel is the number of channels in each input data matrix, 
                    default is 12, which means 4 receivers antenna's T-F data (Amp, Real, Imag)
                in_dim is the number of frequency bins in each input data matrix, default is 512
        output: [B, L * in_dim / 32^2, Embedding], Embedding = 512
        '''

        x = self.conv1(inputs)
        x = self.conv2(x)
        for block in self.ResNet:
            x = block(x)
        x = self.flat(x)
        return x.permute(0, 2, 1).contiguous()
    

class backbone(nn.Module):
    def __init__(self, in_type: str = "1d", cfg: dict = None) -> None:
        super().__init__()
        self.in_type = in_type
        
        self.AutoEncoder_cfg = {
            "in_dim": (cfg["data_size"], cfg["data_size"]),
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        } if in_type == "2d" else {
            "in_dim": cfg["data_size"],
            "in_channel": cfg["in_channels"],
            "drop_path": cfg["drop_path"],
            "with_atten": cfg["Encoder_attention"],
        }
        
        self.encoder = Conv2d_AutoEncoder(**self.AutoEncoder_cfg) \
            if in_type == "2d" else \
                Conv1d_AutoEncoder(**self.AutoEncoder_cfg)
                
        self.embed_dim = self.encoder.channel
    
    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [B, channel, L, in_dim],
        #         B is the batch size
        #         L is the number of frames in each input data matrix, default is 512
        #         channel is the number of channels in each input data matrix,
        #             default is 12, which means 4 receivers antenna's T-F data (Amp, Real, Imag)
        #         in_dim is the number of frequency bins in each input data matrix, default is 512
        
        if self.in_type == "2d":
            return self.encoder(inputs)
        elif self.in_type == "1d":
            b = inputs.shape[0]
            device = inputs.device
            output = torch.stack([self.encoder(inputs[i, ...].permute(1, 0, 2).contiguous()) 
                                  for i in range(b)]).to(device)
            return output


def build_backbone(cfg: dict = None) -> nn.Module:
    return backbone(cfg["in_type"], cfg)        