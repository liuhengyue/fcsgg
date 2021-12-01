"""
Contains some module wrappers, and the implementation of multi-scale batch normalization (MS-BN).
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Shihua Liang https://github.com/sxhxliang/detectron2_backbone"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple, _ntuple
import fvcore.nn.weight_init as weight_init
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    BatchNorm2d, FrozenBatchNorm2d, NaiveSyncBatchNorm,
)



TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

__all__ = ["get_norm", "Conv2dWithPadding", "SeparableConv2d", "MaxPool2d",
           "MultiNormLayer", "MultiNormConvBlock", "MultiNormDeformBlock",
           "MultiscaleFusionLayer", "MultiscaleSum", "DeformConvBlock", "CoordConv", "add_coords"]


def get_norm(norm, out_channels, **kwargs):
    """
    Same with detectron2 get_norm, with kwargs
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        if norm == "FrozenBN":
            return FrozenBatchNorm2d(out_channels)
        if norm == "GN":
            return nn.GroupNorm(32, out_channels)
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class _Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', image_size=None):
        self.padding_mode = padding_mode
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # pading format:
        #     tuple(pad_l, pad_r, pad_t, pad_b) or default
        if padding_mode == 'static_same':
            p = max(kernel_size[0] - stride[0], 0)
            padding = (p // 2, p - p // 2, p // 2, p - p // 2)
        elif padding_mode == 'dynamic_same':
            padding = _pair(0)
        super(_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            input = F.pad(input, expanded_padding, mode='circular')

        elif self.padding_mode == 'dynamic_same':
            ih, iw = input.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                input = F.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        elif self.padding_mode == 'static_same':
            input = F.pad(input, self.padding)
        else:  # default padding
            input = F.pad(input, self.padding)

        return F.conv2d(input,
                        weight, self.bias, self.stride,
                        _pair(0), self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

    def __repr__(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return self.__class__.__name__ + '(' + s.format(**self.__dict__) + ')'


class Conv2dWithPadding(_Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 bias=True, padding_mode='zeros', norm=None, eps=1e-05, momentum=0.1, activation=None):
        super(SeparableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = in_channels
        self.bias = bias
        self.padding_mode = padding_mode

        self.depthwise = Conv2dWithPadding(in_channels, in_channels, kernel_size,
                                stride, padding, dilation, groups=in_channels, bias=False, padding_mode=padding_mode)
        self.pointwise = Conv2dWithPadding(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias, padding_mode=padding_mode)

        self.padding = self.depthwise.padding
        self.norm = None if norm == "" else norm
        if self.norm is not None:
            self.norm = get_norm(norm, out_channels)
            assert self.norm != None
            self.norm.eps = eps
            self.norm.momentum = momentum
        self.activation = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __repr__(self):

        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.pointwise.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.norm is not None:
            s = "  " + s + '\n    norm=' + self.norm.__repr__()
            return self.__class__.__name__ + '(\n  ' + s.format(**self.__dict__) + '\n)'
        else:
            return self.__class__.__name__ + '(' + s.format(**self.__dict__) + ')'


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,
                 padding_mode='static_same'):
        super(MaxPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) or self.kernel_size
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.padding_mode = padding_mode

        if padding_mode == 'static_same':
            p = max(self.kernel_size[0] - self.stride[0], 0)
            # tuple(pad_l, pad_r, pad_t, pad_b)
            padding = (p // 2, p - p // 2, p // 2, p - p // 2)
            self.padding = padding
        elif padding_mode == 'dynamic_same':
            padding = _pair(0)
            self.padding = padding

    def forward(self, input):
        input = F.pad(input, self.padding)
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            _pair(0), self.dilation, self.ceil_mode,
                            self.return_indices)

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}, padding_mode={padding_mode}'.format(**self.__dict__)


class DeformConvBlock(CNNBlockBase):
    """
    modified from detectron2
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=1,
        norm=None,
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated
        self.norm = norm

        assert in_channels == out_channels

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv_offset = Conv2d(
            in_channels,
            offset_channels * deform_num_groups,
            kernel_size=kernel_size,
            stride=stride_3x3,
            padding=padding,
            dilation=dilation,
        )
        self.conv = deform_conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride_3x3,
            padding=padding,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=None,
        )


        for layer in [self.conv]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        if self.deform_modulated:
            offset_mask = self.conv_offset(x)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv(x, offset, mask)
        else:
            offset = self.conv_offset(x)
            out = self.conv(x, offset)
        if self.norm is not None:
            out = self.norm(out)
        out = F.relu_(out)

        return out


class MultiNormLayer(nn.Module):
    def __init__(
            self,
            out_channels,
            output_strides,
            norm="BN",
            bn_momentum=0.01,
            activation=None):
        super().__init__()
        self.norm = norm
        if self.norm in ["BN", "SyncBN"]:
            self.norm_layers = torch.nn.ModuleDict({
                "bn_s" + str(out_stride): get_norm(norm, out_channels, momentum=bn_momentum)
                for out_stride in output_strides})
        else:
            self.norm_layers = get_norm(norm, out_channels)
        self.activation = activation

    def forward(self, x, stride):
        if self.norm in ["BN", "SyncBN"]:
            x = self.norm_layers["bn_s" + str(stride)](x)
        elif self.norm == "GN":
            x = self.norm_layers(x)
        elif self.norm == "":
            pass
        else:
            raise NotImplemented
        if self.activation:
            x = self.activation(x)
        return x

class MultiNormConvBlock(Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            output_strides,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=0,
            norm="BN",
            bn_momentum=0.01,
            activation=None):
        super().__init__(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          bias=not norm,
                          norm=None,
                          activation=None)
        self.norm_layer = MultiNormLayer(out_channels,
                                         output_strides,
                                         norm=norm,
                                         bn_momentum=bn_momentum,
                                         activation=activation)
        # weight_init.c2_msra_fill(self.conv)

    def forward(self, x, stride):
        x = super().forward(x)
        x = self.norm_layer(x, stride)
        return x

class MultiNormDeformBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            output_strides,
            *,
            bottleneck_channels,
            stride=1,
            dilation=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            deform_modulated=False,
            deform_num_groups=1,
            bn_momentum=0.01
    ):
        super().__init__()
        self.deform_modulated = deform_modulated
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        if in_channels != out_channels:
            self.shortcut = MultiNormConvBlock(
                in_channels,
                out_channels,
                output_strides,
                kernel_size=1,
                stride=stride,
                norm=norm,
                bn_momentum=bn_momentum
            )
        else:
            self.shortcut = None

        self.conv1 = MultiNormConvBlock(
            in_channels,
            bottleneck_channels,
            output_strides,
            kernel_size=1,
            stride=stride_1x1,
            norm=norm,
            activation=F.relu_,
            bn_momentum=bn_momentum
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offsets = nn.ModuleDict({
            "offset_s" + str(stride):
            Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
            )
            for stride in output_strides
            })
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=None,
        )

        self.conv2_norm = MultiNormLayer(bottleneck_channels,
                                         output_strides,
                                         norm=norm,
                                         activation=F.relu_,
                                         bn_momentum=bn_momentum)

        self.conv3 = MultiNormConvBlock(
            bottleneck_channels,
            out_channels,
            output_strides,
            kernel_size=1,
            norm=norm,
            bn_momentum=bn_momentum
        )

        weight_init.c2_msra_fill(self.conv2)
        for conv in self.conv2_offsets.values():
            nn.init.constant_(conv.weight, 0)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x, stride):
        out = self.conv1(x, stride)

        if self.deform_modulated:
            offset_mask = self.conv2_offsets["offset_s" + str(stride)](out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offsets["offset_s" + str(stride)](out)
            out = self.conv2(out, offset)
        out = self.conv2_norm(out, stride)

        out = self.conv3(out, stride)

        # shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut(x, stride)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class MultiscaleFusionLayer(nn.Module):
    def __init__(
            self,
            num_branches,
            num_inchannels,
            multi_scale_output=True,
            norm="BN"
    ):
        super(MultiscaleFusionLayer, self).__init__()
        self.norm = norm

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            get_norm(self.norm, num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    get_norm(self.norm, num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    get_norm(self.norm, num_outchannels_conv3x3),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return x

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    z = self.fuse_layers[i][j](x[j])[:, :, : y.shape[2], : y.shape[3]]
                    y = y + z
            x_fuse.append(self.relu(y))

        return x_fuse


class MultiscaleSum(nn.Module):
    def __init__(
            self,
            num_branches,
            aggr_method="max"
    ):
        super(MultiscaleSum, self).__init__()
        self.num_branches = num_branches
        self.aggr_method = aggr_method

    def forward(self, features):
        # sum
        out_feature = features[0]
        for i in range(1, self.num_branches):
            if self.aggr_method == "max":
                out_feature = torch.max(F.interpolate(features[i],
                                                       scale_factor=2 ** i,
                                                       mode='bilinear',
                                                       align_corners=True),
                                         out_feature)
            elif self.aggr_method == "sum":
                out_feature = out_feature + F.interpolate(features[i],
                                                      scale_factor=2 ** i,
                                                      mode='bilinear',
                                                      align_corners=True)
            else:
                raise NotImplementedError()
        # generate different scale
        outputs = [out_feature]
        for i in range(1, self.num_branches):
            outputs.append(F.max_pool2d(out_feature,
                                        kernel_size=2 ** i,
                                        stride=2 ** i))
        return outputs




def add_coords(input_tensor, with_r=False):
    """
    Args:
        input_tensor: shape(batch, channel, x_dim, y_dim)
    """
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat([
        input_tensor,
        xx_channel.type_as(input_tensor),
        yy_channel.type_as(input_tensor)], dim=1)

    if with_r:
        rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)

    return ret


class CoordConv(Conv2d):
    """
    Modified from https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        self.with_r = with_r
        in_size = in_channels + (3 if self.with_r else 2)
        super().__init__(in_size, out_channels, **kwargs)

    def add_coords(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


    def forward(self, x):
        ret = self.add_coords(x)
        ret = super().forward(ret)
        return ret
