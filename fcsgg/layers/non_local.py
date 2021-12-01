"""
Non-local Neural Block Implementation.

This module is originally used for computing attentions between center and RAF features. However, it did not work out.

Copied from https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from .wrappers import get_norm



class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 inter_channels=None,
                 dimension=3,
                 norm=None,
                 sub_sample=True,
                 g_given=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.out_channels = out_channels
        if self.out_channels is None:
            self.out_channels = self.in_channels


        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d(self.out_channels)
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = get_norm(norm, self.out_channels)
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d(self.out_channels)

        if self.in_channels != self.out_channels:
            self.shortcut = conv_nd(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = None
        if not g_given:
            self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if norm is not None:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                        kernel_size=1, stride=1, padding=0),
                bn
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, r=None, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)
        if r is None:
            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
        else:
            g_x = r.view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        if self.shortcut is not None:
            x = self.shortcut(x)
        z = W_y + x
        z = F.relu_(z)

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock1D, self).__init__(in_channels, dimension=1, **kwargs)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock2D, self).__init__(in_channels, dimension=2, **kwargs)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock3D, self).__init__(in_channels, dimension=3, **kwargs)
