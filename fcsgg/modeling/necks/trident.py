"""
TridentConv implementation copied from TridentNet.
"""
__author__ = "Detectron2"
__copyright__ = "Copyright (c) Facebook, Inc."
__credits__ = ["Detectron2"]
__license__ = "Apache-2.0 License"
__version__ = "0.1"
__maintainer__ = "Detectron2"

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import fvcore.nn.weight_init as weight_init
from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.layers import Conv2d, FrozenBatchNorm2d, CNNBlockBase, ShapeSpec
from fcsgg.layers import get_norm
from .necks import NECKS_REGISTRY, Neck

__all__ = ["TridentNeck"]

class TridentConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        paddings=0,
        dilations=1,
        groups=1,
        num_branch=1,
        test_branch_idx=-1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(TridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.dilations)}) == 1

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if inputs[0].numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    inputs[0].shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [input[0].shape[0], self.weight.shape[0]] + output_shape
            return [_NewEmptyTensorOp.apply(input, output_shape) for input in inputs]

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, self.stride, padding, dilation, self.groups)
                for input, dilation, padding in zip(inputs, self.dilations, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.stride,
                    self.paddings[self.test_branch_idx],
                    self.dilations[self.test_branch_idx],
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", num_branch=" + str(self.num_branch)
        tmpstr += ", test_branch_idx=" + str(self.test_branch_idx)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", paddings=" + str(self.paddings)
        tmpstr += ", dilations=" + str(self.dilations)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr


class TridentBottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        bn_momentum=0.01,
        stride_in_1x1=False,
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
    ):
        """
        Args:
            num_branch (int): the number of branches in TridentNet.
            dilations (tuple): the dilations of multiple branches in TridentNet.
            concat_output (bool): if concatenate outputs of multiple branches in TridentNet.
                Use 'True' for the last trident block.
        """
        super().__init__(in_channels, out_channels, stride)

        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels, momentum=bn_momentum)
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels, momentum=bn_momentum),
        )

        self.conv2 = TridentConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            paddings=dilations,
            bias=False,
            groups=num_groups,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=get_norm(norm, bottleneck_channels, momentum=bn_momentum),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels, momentum=bn_momentum),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch
        out = [self.conv1(b) for b in x]
        out = [F.relu_(b) for b in out]

        out = self.conv2(out)
        out = [F.relu_(b) for b in out]

        out = [self.conv3(b) for b in out]

        if self.shortcut is not None:
            shortcut = [self.shortcut(b) for b in x]
        else:
            shortcut = x

        out = [out_b + shortcut_b for out_b, shortcut_b in zip(out, shortcut)]
        out = [F.relu_(b) for b in out]
        if self.concat_output:
            out = torch.cat(out)
        return out

def make_trident_stage(block_class, num_blocks, first_stride, stage_out_channels, **kwargs):
    """
    Create a resnet stage by creating many blocks for TridentNet.
    """
    blocks = []
    for i in range(num_blocks - 1):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    kwargs["out_channels"] = stage_out_channels
    blocks.append(block_class(stride=1, concat_output=False, **kwargs))
    return blocks

@NECKS_REGISTRY.register()
class TridentNeck(Neck):
    def __init__(self, cfg, input_shape):
        super(TridentNeck, self).__init__(cfg, input_shape)
        neck_cfg = cfg.MODEL.NECKS
        self.in_features = neck_cfg.IN_FEATURES
        # cat features
        self.in_channels = sum([input_shape[k].channels for k in self.in_features])
        self.out_channels = neck_cfg.OUT_CHANNELS[0]
        self.block_out_channels = neck_cfg.TRIDENT.BLOCK_OUT_CHANNELS
        self.bottleneck_channels = neck_cfg.TRIDENT.BOTTLENECK_CHANNELS

        upsample_mode = cfg.MODEL.NECKS.UPSAMPLE_MODE
        self.upsample_mode = {"mode": upsample_mode, "align_corners": None} if upsample_mode == "nearest" \
            else {"mode": upsample_mode, "align_corners": True}

        norm = neck_cfg.NORM
        branch_dilations = neck_cfg.TRIDENT.BRANCH_DILATIONS
        num_branches = neck_cfg.TRIDENT.NUM_BRANCH
        self._out_feature_strides = {f"d{d}": d for d in branch_dilations}
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: self.out_channels for k in self._out_features}

        stage_kargs = {
            "first_stride": 2,
            "in_channels": self.in_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "out_channels": self.block_out_channels,
            "num_groups": 1,
            "norm": norm,
            "stride_in_1x1": False,
            "block_class": TridentBottleneckBlock,
            "num_branch": num_branches,
            "dilations": branch_dilations,
            "num_blocks": 2,
            "bn_momentum": neck_cfg.MOMENTUM
        }
        blocks = make_trident_stage(stage_out_channels=self.out_channels, **stage_kargs)
        self.trident_convs = nn.Sequential(*blocks)


    def forward(self, features: Dict[str, torch.Tensor]):
        # features of different scale, first is the largest
        features = [features[f] for f in self.in_features]
        outputs = [features[0]]
        target_size = features[0].size()[2:]
        for i in range(1, len(features)):
            outputs.append(F.interpolate(features[i],
                                         size=target_size,
                                         mode=self.upsample_mode["mode"],
                                         align_corners=self.upsample_mode["align_corners"]))
        # cat features from hrnet
        outputs = torch.cat(outputs, dim=1)

        # trident blocks
        outputs = self.trident_convs(outputs)
        return dict(zip(self._out_features, outputs))


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }