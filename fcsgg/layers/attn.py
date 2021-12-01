"""
implementations of attention modules of DANet and CBAM.

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Jun Fu", "Sanghyun Woo"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from detectron2.layers import Conv2d
from fcsgg.layers import get_norm
torch_ver = torch.__version__[:3]

__all__ = ['DA_Module', 'CBAM']


class DA_Module(nn.Module):
    """ Modified from the dualattention"""
    def __init__(self, in_dim, out_dim, down_ratio=1, conv_norm='GN', bn_momentum=0.01):
        super(DA_Module, self).__init__()
        self.down_ratio = down_ratio
        self.channel_in = in_dim
        self.channel_out = out_dim
        self.channel_interm = out_dim
        stride = self.down_ratio
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim,
                                    kernel_size=3, stride=stride, padding=2, dilation=2)
        self.obj_key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim,
                                      kernel_size=3, stride=stride, padding=2, dilation=2)
        self.rel_key_conv = Conv2d(in_channels=in_dim, out_channels=out_dim,
                                      kernel_size=3, stride=stride, padding=2, dilation=2)
        # post conv after attention
        self.value_conv = nn.Conv2d(in_channels=2 * out_dim, out_channels= 2 * out_dim, kernel_size=1)
        self.gamma_spatial = nn.Parameter(torch.zeros(1))
        self.gamma_channel = nn.Parameter(torch.zeros(1))

        # self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input center feature maps ( B X C X H X W)
                y : input RAF feature maps ( B X P X H/2 X W/2) or ( B X C X H X W)
            returns :
                out : attention value + input feature
                feature maps( B X C' X H X W)
                attention: B X (HxW) X (HxW)
        """
        shortcut = y
        m_batchsize, C, height, width = x.size()
        # (150, HW/4)
        proj_query = self.query_conv(x).view(m_batchsize, self.channel_in, -1)
        # (150, HW/4)
        obj_proj_key = self.obj_key_conv(x).view(m_batchsize, self.channel_in, -1)
        # (50, HW/4)
        rel_proj_key = self.rel_key_conv(x).view(m_batchsize, self.channel_out, -1)
        # (HW) x (HW)
        spatial_attention = F.softmax(torch.bmm(proj_query.permute(0, 2, 1), obj_proj_key), dim=-1).permute(0, 2, 1)
        # 50 x 150
        channel_attention = F.softmax(torch.bmm(rel_proj_key, proj_query.permute(0, 2, 1)), dim=-1)
        y  = y.view(m_batchsize, self.channel_out, 2, -1)
        # (B, 50, 2, HW/4)
        out = torch.stack((torch.bmm(y[:, :, 0, :], spatial_attention),
                           torch.bmm(y[:, :, 1, :], spatial_attention)),
                          dim=2) * self.gamma_spatial
        # (B, 50, HW)
        x = torch.bmm(channel_attention, x.view(m_batchsize, C, -1)) * self.gamma_channel
        # x = x.view(m_batchsize, -1, height, width)
        x = x.view(m_batchsize, self.channel_out, height, width).repeat_interleave(2, dim=1)
        out = out.view(m_batchsize, -1, height//self.down_ratio, width//self.down_ratio) + shortcut
        if self.down_ratio > 1:
            out = F.interpolate(out,
                                size=(height, width),
                                mode="nearest",
                                align_corners=None)
        out += x
        out = self.value_conv(out)
        # out = torch.tanh(out)
        return out


"""
Following are implementation of "Dual Attention Network for Scene Segmentation(CVPR2019)"

Credit: https://github.com/junfu1115/DANet/blob/master/encoding/nn/da_att.py

"""

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

"""
Following are implementation of "CBAM: Convolutional Block Attention Module (ECCV2018)"
 
Credit: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py

"""



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, conv_norm="SyncBN", bn_momentum=0.01):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv2d(2, 1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              dilation=1,
                              bias=not conv_norm,
                              activation=None,
                              norm=get_norm(conv_norm, 1, momentum=bn_momentum))
        init.kaiming_normal(self.spatial.weight, mode='fan_out')

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max'],
                 no_spatial=False,
                 conv_norm="SyncBN",
                 bn_momentum=0.01):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(conv_norm=conv_norm, bn_momentum=bn_momentum)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out