"""
Implementations of HRNet FPN.

# Modified from: https://github.com/HRNet/HRNet-MaskRCNN-Benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/hrfpn.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["https://github.com/HRNet"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .necks import NECKS_REGISTRY, Neck
from detectron2.layers import Conv2d
# from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from fcsgg.layers import get_norm, DeformConvBlock
@NECKS_REGISTRY.register()
class HRFPN(Neck):

    def __init__(self, cfg, input_shape, deform_on=False):
        super(HRFPN, self).__init__(cfg, input_shape)

        config = cfg.MODEL.NECKS
        bn_momentum = 0.01
        self.norm = config.NORM
        self.pooling_type = config.POOLING
        self.in_features = config.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]

        self.out_channels = config.OUT_CHANNELS[0]
        self.num_ins = len(self.in_channels)
        assert isinstance(self.in_channels, (list, tuple))
        # in_strides = [input_shape[k].stride for k in self.in_features]
        out_strides = config.OUT_STRIDES
        conv_stride = config.CONV_STRIDE
        self.num_outs = len(out_strides)
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in out_strides}
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: self.out_channels for k in self._out_features}
        self._size_divisibility = self._out_feature_strides[self._out_features[-1]]
        #
        self.reduction_conv = Conv2d(in_channels=sum(self.in_channels),
                                       out_channels=self.out_channels,
                                       kernel_size=1,
                                       norm=get_norm(self.norm, self.out_channels, momentum=bn_momentum))
        self.fpn_conv = nn.ModuleList()

        # determine conv type
        if deform_on:
            conv = DeformConvBlock
        else:
            conv = Conv2d
        for i in range(self.num_outs):
            self.fpn_conv.append(conv(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=conv_stride,
                padding=1,
                norm=get_norm(self.norm, self.out_channels, momentum=bn_momentum)
            ))
        if self.pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, inputs):
        inputs = [inputs[f] for f in self.in_features]
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear', align_corners=True))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))
        return dict(zip(self._out_features, outputs))

@NECKS_REGISTRY.register()
class DualHRFPN(Neck):

    def __init__(self, cfg, input_shape):
        super(DualHRFPN, self).__init__(cfg, input_shape)
        self.num_branchs = 2
        deform_on_per_branch = cfg.MODEL.NECKS.DEFORM_ON_PER_STAGE
        assert len(deform_on_per_branch) == self.num_branchs
        self.fpns = nn.ModuleList([
            HRFPN(cfg, input_shape, deform_on=deform_on_per_branch[i])
            for i in range(self.num_branchs)
        ])
        config = cfg.MODEL.NECKS
        out_strides = config.OUT_STRIDES
        self._out_feature_strides = {f"p{int(math.log2(s))}_0": s for s in out_strides}
        self._out_feature_strides.update({f"p{int(math.log2(s))}_1": s for s in out_strides})
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: config.OUT_CHANNELS[0] for k in self._out_features}

    def forward(self, inputs):
        outputs = []
        for i in range(self.num_branchs):
            outputs += self.fpns[i](inputs).values()
        return dict(zip(self._out_features, outputs))