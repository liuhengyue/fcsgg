"""
Implementations of necks (intermediate branches transforming backbone features).

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import inspect
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, CNNBlockBase, ConvTranspose2d, FrozenBatchNorm2d
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from detectron2.layers.deform_conv import DeformConv, ModulatedDeformConv
from detectron2.layers import ShapeSpec, Conv2d, get_norm

from fcsgg.layers import add_coords


NECKS_REGISTRY = Registry("NECKS")
NECKS_REGISTRY.__doc__ = """
Registry for neck modules in a single-stage model (or even for generalized R-CNN model).
A neck takes feature maps and applies transformation of the feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Neck`.
"""

logger = logging.getLogger(__name__)


class Neck(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.freeze_necks = cfg.MODEL.NECKS.FREEZE

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self):
        if self.freeze_necks:
            # freeze all weights of a neck
            for p in self.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class UpSampleBlock(CNNBlockBase):

    def __init__(
        self, in_planes,
        out_planes, *, bottleneck_channels,
        deconv_kernel=3, deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, deform_num_groups=1, deform_modulated=True,
        norm="BN"
    ):
        super(UpSampleBlock, self).__init__(in_channels=in_planes,
                                            out_channels=out_planes,
                                            stride=deconv_stride)

        self.dcn = DeformBottleneckBlock(
                in_planes, out_planes, bottleneck_channels=bottleneck_channels,
                deform_modulated=deform_modulated,
                deform_num_groups=deform_num_groups
            )

        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = get_norm(norm, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dcn(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

@NECKS_REGISTRY.register()
class Res5UpSampleNeck(nn.Module):

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_features = cfg.MODEL.NECKS.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]
        self.out_channels = cfg.MODEL.NECKS.OUT_CHANNELS
        self.num_groups = cfg.MODEL.NECKS.NUM_GROUPS
        self.deform_on_per_stage = cfg.MODEL.NECKS.DEFORM_ON_PER_STAGE
        self.deconv_kernel_sizes = cfg.MODEL.NECKS.DECONV_KERNEL_SIZES
        self.deform_modulated = cfg.MODEL.NECKS.DEFORM_MODULATED
        self.deform_num_groups = cfg.MODEL.NECKS.DEFORM_NUM_GROUPS
        self.width_per_group = cfg.MODEL.NECKS.WIDTH_PER_GROUP
        self.bottleneck_channels = self.num_groups * self.width_per_group
        self._out_features = cfg.MODEL.NECKS.OUT_FEATURES

        assert len(self.deform_on_per_stage) == len(self.out_channels)
        assert len(self.in_features) == 1 and len(self.in_channels) == 1

        self.stages_and_names = []
        self._out_feature_strides ={}
        self._out_feature_channels = {}

        blocks = self._make_blocks()
        current_stride = [input_shape[k].stride for k in self.in_features][0]
        for i, block in enumerate(blocks):
            name = "upsample_" + str(i)
            self.add_module(name, block)
            self.stages_and_names.append((block, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * block.stride
            )
            self._out_feature_channels[name] = block.out_channels


    def _make_blocks(self):
        blocks = []
        in_channels = self.in_channels[0]
        for i, deform_on in enumerate(self.deform_on_per_stage):
            out_channel = self.out_channels[i]
            if deform_on:
                # this block will upsample x2
                block = UpSampleBlock(in_channels, out_channel,
                                      bottleneck_channels=self.bottleneck_channels,
                                      deconv_kernel=self.deconv_kernel_sizes[i],
                                      deform_num_groups=self.deform_num_groups)
                blocks.append(block)
                in_channels = out_channel

        return blocks





    def forward(self, features: Dict[str, torch.Tensor]):
        features = [features[f] for f in self.in_features][0]
        outputs = {}
        for stage, name in self.stages_and_names:
            features = stage(features)
            if name in self._out_features:
                outputs[name] = features
        return outputs


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class IDAUp(nn.Module):

    def __init__(self, out_channels, channels, up_f, *,
                 bottleneck_channels, deform_modulated=True, norm="BN"):
        super(IDAUp, self).__init__()
        self.projections = []
        self.up_convs = []
        self.nodes = []
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformBottleneckBlock(c, out_channels, bottleneck_channels=bottleneck_channels,
                                         deform_modulated=deform_modulated, deform_num_groups=1,
                                         norm=norm)
            node = DeformBottleneckBlock(out_channels, out_channels, bottleneck_channels=bottleneck_channels,
                                         deform_modulated=deform_modulated, deform_num_groups=1,
                                         norm=norm)

            up = nn.ConvTranspose2d(out_channels, out_channels, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=out_channels, bias=False)
            self.fill_up_weights(up)

            self.add_module('proj_' + str(i), proj)
            self.add_module('up_' + str(i), up)
            self.add_module('node_' + str(i), node)
            self.projections.append(proj)
            self.up_convs.append(up)
            self.nodes.append(node)

            # setattr(self, 'proj_' + str(i), proj)
            # setattr(self, 'up_' + str(i), up)
            # setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(endp - startp - 1):
            upsample = self.up_convs[i]
            project = self.projections[i]
            layers[i + startp + 1] = upsample(project(layers[i + startp + 1]))
            node = self.nodes[i]
            layers[i + startp + 1] = node(layers[i + startp + 1] + layers[i + startp])
    # def forward(self, layers, startp, endp):
    #     for i in range(startp + 1, endp):
    #         upsample = getattr(self, 'up_' + str(i - startp))
    #         project = getattr(self, 'proj_' + str(i - startp))
    #         layers[i] = upsample(project(layers[i]))
    #         node = getattr(self, 'node_' + str(i - startp))
    #         layers[i] = node(layers[i] + layers[i - 1])

    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class DLAUp(nn.Module):
    def __init__(self, channels, scales, *,
                 in_channels=None, bottleneck_channels,
                 norm="BN"):
        super(DLAUp, self).__init__()
        self.ida_modules = []
        if in_channels is None:
            in_channels = channels
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            ida = IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j], bottleneck_channels=bottleneck_channels,
                        norm=norm)
            self.add_module('ida_{}'.format(i), ida)
            self.ida_modules.append(ida)
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - 1):
            ida = self.ida_modules[i]
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

@NECKS_REGISTRY.register()
class DLAUpSampleNeck(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_features = cfg.MODEL.NECKS.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]
        self.num_groups = cfg.MODEL.NECKS.NUM_GROUPS
        self.deform_modulated = cfg.MODEL.NECKS.DEFORM_MODULATED
        self.deform_num_groups = cfg.MODEL.NECKS.DEFORM_NUM_GROUPS
        self.width_per_group = cfg.MODEL.NECKS.WIDTH_PER_GROUP
        self.bottleneck_channels = self.num_groups * self.width_per_group
        self.down_ratio = cfg.MODEL.HEADS.OUTPUT_STRIDE
        self.norm = cfg.MODEL.NECKS.NORM

        # only output one feature
        self._out_features = ["dla_fused"]
        out_channel = self.in_channels[0]
        self._out_feature_channels = {"dla_fused": out_channel}
        self._out_feature_strides = {"dla_fused": self.down_ratio}


        scales = [2 ** i for i in range(len(self.in_channels))]
        self.dla_up = DLAUp(self.in_channels[:], scales,
                            bottleneck_channels=self.bottleneck_channels,
                            norm=self.norm)

        scales = [2 ** i for i in range(len(self.in_channels) - 1)]
        self.ida_up = IDAUp(out_channel, self.in_channels[:-1],
                            scales, bottleneck_channels=self.bottleneck_channels,
                            norm=self.norm)

    def forward(self, features: Dict[str, torch.Tensor]):
        features = [features[f] for f in self.in_features]
        features = self.dla_up(features)[:-1]
        self.ida_up(features, 0, len(features))
        # return the last output for heads
        return {"dla_fused": features[-1]}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    def freeze(self):
        return self

class DeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=2,
                 leakyReLU=False, norm="BN"):
        super(DeConv2d, self).__init__()
        # deconv basic config
        if ksize == 4:
            padding = 1
            output_padding = 0
        elif ksize == 3:
            padding = 1
            output_padding = 1
        elif ksize == 2:
            padding = 0
            output_padding = 0

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               ksize, stride=stride,
                               padding=padding, output_padding=output_padding,
                               bias=not norm),
            get_norm(norm, out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

@NECKS_REGISTRY.register()
class FPNUpSampleNeck(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_features = cfg.MODEL.NECKS.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]
        self.num_groups = cfg.MODEL.NECKS.NUM_GROUPS
        self.deform_modulated = cfg.MODEL.NECKS.DEFORM_MODULATED
        self.deform_num_groups = cfg.MODEL.NECKS.DEFORM_NUM_GROUPS
        self.width_per_group = cfg.MODEL.NECKS.WIDTH_PER_GROUP
        self.bottleneck_channels = self.num_groups * self.width_per_group
        self.down_ratio = cfg.MODEL.HEADS.OUTPUT_STRIDE
        self.norm = cfg.MODEL.NECKS.NORM
        self.freeze_necks = cfg.MODEL.NECKS.FREEZE
        self._out_features = ["fpn_fused"]
        self._out_feature_channels = {"fpn_fused": self.in_channels[0]}
        self._out_feature_strides = {"fpn_fused": self.down_ratio}
        # keep p2 and upsample from p5
        self.stages = []
        # all in_channels are 256
        for i, in_channel in enumerate(self.in_channels[1:]):
            up_conv = DeConv2d(in_channel, in_channel, ksize=2, stride=2, norm=self.norm)
            name = "upsample_" + str(i)
            self.add_module(name, up_conv)
            self.stages.append(up_conv)

    def forward(self, features: Dict[str, torch.Tensor]):
        features = [features[f] for f in self.in_features][::-1] # [p5, p4, p3, p2]
        for i in range(len(features) - 1):
            features[i+1] = features[i+1] + self.stages[i](features[i])
        return {self._out_features[0]: features[-1]}


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self):
        if self.freeze_necks:
            # freeze all weights of a neck
            for p in self.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


@NECKS_REGISTRY.register()
class ConcatNeck(Neck):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.in_features = cfg.MODEL.NECKS.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]
        self.num_groups = cfg.MODEL.NECKS.NUM_GROUPS
        self.deform_modulated = cfg.MODEL.NECKS.DEFORM_MODULATED
        self.deform_num_groups = cfg.MODEL.NECKS.DEFORM_NUM_GROUPS
        self.width_per_group = cfg.MODEL.NECKS.WIDTH_PER_GROUP
        self.bottleneck_channels = self.num_groups * self.width_per_group
        self.down_ratio = cfg.MODEL.HEADS.OUTPUT_STRIDE
        self.norm = cfg.MODEL.NECKS.NORM
        self.fuse_method = "cat"
        self._out_features = [self.fuse_method]
        self._out_feature_channels = {self.fuse_method: sum(self.in_channels)}
        self._out_feature_strides = {self.fuse_method: self.down_ratio}
        upsample_mode = cfg.MODEL.NECKS.UPSAMPLE_MODE
        self.upsample_mode = {"mode": upsample_mode, "align_corners": None} if upsample_mode == "nearest" \
            else {"mode": upsample_mode, "align_corners": True}
        # for coordconv
        self.add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD

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

        outputs = torch.cat(outputs, dim=1)
        if self.add_coord:
            outputs = add_coords(outputs)
        return dict(zip(self._out_features, [outputs]))


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



def build_necks(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    use_neck = cfg.MODEL.NECKS.ENABLED
    if use_neck:
        name = cfg.MODEL.NECKS.NAME
        return NECKS_REGISTRY.get(name)(cfg, input_shape).freeze()
    else:
        return None