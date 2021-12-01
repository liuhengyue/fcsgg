"""
Implementations of EfficientNet Bi-directional FPN.
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

import fvcore.nn.weight_init as weight_init
from detectron2.modeling.backbone import Backbone, build_resnet_backbone
# from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec, get_norm

from fcsgg.layers import Conv2dWithPadding, SeparableConv2d, MaxPool2d, Swish, MemoryEfficientSwish
# from .efficientnet import build_efficientnet_backbone


class Attention(nn.Module):
    def __init__(self, num_inputs, eps=1e-3):
        super().__init__()
        self.num_inputs = num_inputs
        self.atten_w = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == self.num_inputs
        atten_w = F.relu(self.atten_w)
        return sum(x_ * w for x_, w in zip(inputs, atten_w)) / (atten_w.sum() + self.eps)

    def __repr__(self):
        s = ('num_inputs={num_inputs}'
             ', eps={eps}')
        return self.__class__.__name__ + '(' + s.format(**self.__dict__) + ')'


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels, num_stages=4, lateral=False, attention=True, norm='', epsilon=1e-4,
                 top_block=None):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon
        self.attention = attention
        self.lateral = lateral
        self.num_stages = num_stages - 1
        mom = 0.01
        eps = 1e-3

        # Conv layers
        self.conv_ups, self.conv_downs = [], []
        for i in range(self.num_stages):
            block = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
            self.conv_ups.append(block)
            self.add_module("conv_up_" + str(i + 3), block)

            block = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                    momentum=mom, eps=eps)
            self.conv_downs.append(block)
            self.add_module("conv_down_" + str(i + 4), block)
        if attention:
            self.init_attention_weights()
        self._downsample = MaxPool2d(3, 2)
        self._swish = MemoryEfficientSwish()

    def init_attention_weights(self):
        # top-down Weight
        for i in range(self.num_stages):
            weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.__setattr__("p" + str(i + 3) + "_w1", weight)
            num_tensors = 3 if i < self.num_stages - 1 else 2
            weight = nn.Parameter(torch.ones(num_tensors, dtype=torch.float32), requires_grad=True)
            self.__setattr__("p" + str(i + 4) + "_w2", weight)

    def _weight_act(self, weight):
        weight = F.relu(weight)
        return weight / (torch.sum(weight, dim=0) + self.epsilon)

    def _attention(self, weight, inputs):
        assert isinstance(inputs, list) and len(inputs) == len(weight)
        return sum(x_ * w for x_, w in zip(inputs, weight))

    def _upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def _feature_funsion(self, cur_feature, top_feature, indice=-1):
        top_feature = self._upsample(top_feature)
        if self.attention and indice > 0:
            weight = getattr(self, "p{}_w1".format(indice))
            return self._attention(weight, [cur_feature, top_feature])
        else:
            return cur_feature + top_feature

    def _feature_funsion2(self, skip_feature, cur_feature, bottom_feature, indice=-1):
        _downsample = self._downsample(bottom_feature)
        if self.attention and indice > 0:
            if isinstance(skip_feature, torch.Tensor):
                weight = getattr(self, "p{}_w2".format(indice))
                return self._attention(weight, [skip_feature, cur_feature, _downsample])
            else:
                weight = getattr(self, "p{}_w2".format(indice))
                return self._attention(weight, [cur_feature, _downsample])
        else:
            if isinstance(skip_feature, torch.Tensor):
                return skip_feature + cur_feature + _downsample
            else:
                return cur_feature + _downsample

    def _forward_up(self, inputs):

        inputs = inputs[::-1] # reversed [p7_in, p6_in, ..., p3_in]
        outputs = []
        x = inputs[0]
        outputs.append(x)
        for i in range(len(inputs) - 1):
            ind = self.num_stages + 2 - i
            x = self.conv_ups[-1 + i](self._swish(self._feature_funsion(inputs[i+1], x, ind)))
            outputs.append(x)
        return outputs[::-1]
        # p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        # # print( p3_in.shape, p4_in.shape, p5_in.shape, p6_in.shape, p7_in.shape)
        # # Connections for P6_0 and P7_0 to P6_1 respectively
        # p6_up = self.conv6_up(self._swish(self._feature_funsion(p6_in, p7_in, 6)))
        # p5_up = self.conv5_up(self._swish(self._feature_funsion(p5_in, p6_up, 5)))
        # p4_up = self.conv4_up(self._swish(self._feature_funsion(p4_in, p5_up, 4)))
        # p3_up = self.conv3_up(self._swish(self._feature_funsion(p3_in, p4_up, 3)))
        # return p3_up, p4_up, p5_up, p6_up, p7_in

    def _forward_down(self, laterals, up_features, skip_features):
        # _, p4_in, p5_in, p6_in, p7_in = laterals
        # p3_up, p4_up, p5_up, p6_up, _ = up_features
        #
        # if self.lateral:
        #     p4_in, p5_in = skip_features
        #
        # p4_out = self.conv4_down(self._swish(
        #     self._feature_funsion2(p4_in, p4_up, p3_up, 4)))
        # p5_out = self.conv5_down(self._swish(
        #     self._feature_funsion2(p5_in, p5_up, p4_out, 5)))
        # p6_out = self.conv6_down(self._swish(
        #     self._feature_funsion2(p6_in, p6_up, p5_out, 6)))
        # p7_out = self.conv7_down(self._swish(
        #     self._feature_funsion2(None, p7_in, p6_out, 7)))
        # return p3_up, p4_out, p5_out, p6_out, p7_out

        if self.lateral:
            laterals[1] = skip_features[0]
            laterals[2] = skip_features[1]
        outputs = []
        x = up_features[0]
        outputs.append(x)
        for i in range(len(up_features) - 1):
            ind = 4 + i
            lateral_feat = laterals[i+1] if i < len(up_features) - 2 else None
            x = self.conv_downs[i](self._swish(self._feature_funsion2(lateral_feat, up_features[i+1], x, ind)))
            outputs.append(x)
        return outputs

    def forward(self, inputs):
        laterals, skip_features = inputs
        up_features = self._forward_up(laterals)

        return self._forward_down(laterals, up_features, skip_features), None

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class BeforeBiFPNLayer(nn.Module):
    def __init__(self, out_channels, in_channels=None,
                 kernel_size=1,
                 epsilon=1e-4,
                 top_block=None,
                 norm="BN"):
        super(BeforeBiFPNLayer, self).__init__()
        self.epsilon = epsilon
        mom = 0.01
        eps = 1e-3
        self.norm = nn.SyncBatchNorm if norm == "SyncBN" else nn.BatchNorm2d
        self.laterals = []
        self.shortcuts = []
        start_stage = 6 - len(in_channels)
        for i, in_channel in enumerate(in_channels):
            block = nn.Sequential(
            Conv2dWithPadding(in_channel, out_channels, kernel_size, stride=1, padding_mode='static_same'),
            self.norm(out_channels, momentum=mom, eps=eps))
            self.laterals.append(block)
            self.add_module("lateral_" + str(i + start_stage), block)

        self.top_block = top_block
        self.num_stages = len(self.laterals)

        # backbone skip connection
        for i in range(1, 3):
            block = nn.Sequential(
            Conv2dWithPadding(in_channels[i], out_channels, 1, stride=1, padding_mode='static_same'),
            self.norm(out_channels, momentum=mom, eps=eps))
            self.shortcuts.append(block)
            self.add_module("shortcut_" + str(i + start_stage), block)

    def forward(self, inputs):
        lateral_outputs, shortcuts = [], []
        for i in range(len(inputs)):
            lateral_outputs.append(self.laterals[i](inputs[i]))
            if i > 0 and i < 3:
                shortcuts.append(self.shortcuts[i - 1](inputs[i]))
        if self.top_block:
            lateral_outputs += self.top_block(inputs[-1])
        return lateral_outputs, shortcuts


class BiFPN(Backbone):
    """
    This module implements Bi-Derectional Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self,
                 bottom_up,
                 in_features,
                 out_channels,
                 fpn_repeat,
                 norm="SyncBN",
                 top_block=None,
                 fuse_type="sum"):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [bottom_up._out_feature_strides[f] for f in in_features]
        in_channels = [bottom_up._out_feature_channels[f] for f in in_features]

        self.in_features = in_features
        self.bottom_up = bottom_up

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in in_strides}

        # top block output feature maps.
        last_stage = int(math.log2(in_strides[-1]))
        extra_levels = top_block.num_levels if top_block else 0
        for s in range(last_stage, last_stage + extra_levels):
            in_strides.append(2 ** (s + 1))
            in_channels.append(out_channels)
            self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        _assert_strides_are_log2_contiguous(in_strides)
        self.before_bifpn = BeforeBiFPNLayer(out_channels, in_channels, top_block=top_block, norm=norm)
        layers = []
        num_stages = self.before_bifpn.num_stages
        for i in range(fpn_repeat):
            lateral = True if i == 0 else False
            if i > 0: lateral = False
            layers.append(BiFPNLayer(out_channels, num_stages, norm=norm, lateral=lateral))
        self.bifpn = nn.Sequential(*layers)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = self._out_feature_strides[self._out_features[-1]]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        features = [bottom_up_features[f] for f in self.in_features]

        lateral_features, skip_features = self.before_bifpn(features)

        features, _ = self.bifpn((lateral_features, skip_features))
        assert len(self._out_features) == len(features)
        return dict(zip(self._out_features, features))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], \
            f"Strides {stride} {strides[i - 1]} are not log2 contiguous"


class ResampleFeature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm):
        super().__init__()
        self.conv = Conv2dWithPadding(in_channels, out_channels, kernel_size=1, stride=1, padding_mode="static_same")
        self.norm = get_norm(norm, out_channels) if norm != '' else lambda x: x
        self.resample = MaxPool2d(kernel_size=3, stride=2, padding_mode="static_same")

        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        return self.resample(self.norm(self.conv(x)))


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, norm=''):
        super().__init__()
        self.num_levels = 2
        self.p6 = ResampleFeature(in_channels, out_channels, 1, norm=norm)
        self.p7 = MaxPool2d(kernel_size=3, stride=2, padding_mode="static_same")
        # ResampleFeature(out_channels, out_channels, 1, norm=norm)

    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(p6)
        return [p6, p7]

class LastLevelP6(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, norm=''):
        super().__init__()
        self.num_levels = 1
        self.p6 = ResampleFeature(in_channels, out_channels, 1, norm=norm)

    def forward(self, p5):
        p6 = self.p6(p5)
        return [p6]

@BACKBONE_REGISTRY.register()
def build_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()[in_features[-1]].channels
    if len(in_features) == 4:
        top_block = None
    else:
        top_block = LastLevelP6P7(in_channels_p6p7,
                                 out_channels,
                                 cfg.MODEL.FPN.NORM)
    backbone = BiFPN(bottom_up=bottom_up,
                     in_features=in_features,
                     out_channels=out_channels,
                     fpn_repeat=cfg.MODEL.FPN.REPEAT,
                     norm=cfg.MODEL.FPN.NORM,
                     top_block=top_block,
                     fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone