"""
Implementations of a Hourglass network that has a backbone encoder with two decoders.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = []
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import math
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, Conv2d, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.resnet import BottleneckBlock


class TriadEncoder(nn.Module):
    def __init__(self, num_blocks, in_planes, out_planes, depth, num_branches=2, norm="BN"):
        super(TriadEncoder, self).__init__()
        self.depth = depth
        assert len(out_planes) == depth + 1
        self.out_planes = out_planes
        self.norm = norm
        self.num_branches = num_branches
        self.hg = self._make_hour_glass(num_blocks, in_planes, out_planes, depth)

    def _make_residual(self, num_blocks, in_planes, out_planes):
        layers = []
        layers.append(BottleneckBlock(in_planes,
                                      out_planes,
                                      bottleneck_channels=out_planes // 2,
                                      norm=self.norm))
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_planes,
                                            out_planes,
                                            bottleneck_channels=out_planes // 2,
                                            norm=self.norm))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, num_blocks, in_planes, out_planes, depth):
        """
        Revisit Hourglass design:
            first block: the residual block from c1 to c1_0
            second block: c1 after maxpool to c2
            thrid block for hourglass: conv blocks before upsample c2 to c2_up
        |-------- c1_0 -------- |
        |                       |
        |                       |
        |                       |
        c1 -> c2 ... c2_up ---> + c1_out
        """
        hg = []
        for i in range(depth):
            res = []

            # first residual, possibly more residuals
            res += [self._make_residual(num_blocks, in_planes, in_planes) for _ in range(self.num_branches)]
            # second downsample stream
            res.append(self._make_residual(num_blocks, in_planes, out_planes[i + 1]))
            in_planes = out_planes[i + 1]
            if i == depth - 1:
                res.append(self._make_residual(num_blocks, in_planes, in_planes))
            hg.append(nn.ModuleList(res))
        # inverse it since we starts with the lowest depth
        return nn.ModuleList(hg[::-1])

    def _hour_glass_forward(self, n, x):
        res = []
        for i in range(self.num_branches):
            res.append(self.hg[n][i](x))
        low1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        low1 = self.hg[n][self.num_branches](low1)
        if n == 0:
            # the smallest feature maps does not need skip connections
            low1 = self.hg[n][self.num_branches + 1](low1)
        # list and tensor
        return res, low1

    def forward(self, x):
        res_features = []
        for depth in reversed(range(self.depth)):
            res, x = self._hour_glass_forward(depth, x)
            res_features.append(res)
        # make resfeatures from small to large
        res_features = res_features[::-1]
        return x, res_features

class TriadDecoder(nn.Module):
    def __init__(self, num_blocks, in_planes, out_planes, depth, expansion=2, norm="BN",
                 intermediate_output=False, output_stride=1):
        super(TriadDecoder, self).__init__()
        self.depth = depth
        self.norm = norm
        self.expansion = expansion
        self.output_stride = output_stride
        self.stop_upsample_idx = depth - int(math.log2(self.output_stride))
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(num_blocks, in_planes, out_planes, depth)
        self.intermediate_output = intermediate_output
        if self.intermediate_output:
            self.bridge = self._make_decoder_to_decoder_residual(num_blocks, in_planes, out_planes, depth)

    def _make_residual(self, num_blocks, in_planes, out_planes):
        layers = []
        layers.append(BottleneckBlock(in_planes,
                                      out_planes,
                                      bottleneck_channels=out_planes // self.expansion,
                                      norm=self.norm))
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_planes,
                                          out_planes,
                                          bottleneck_channels=out_planes // self.expansion,
                                          norm=self.norm))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, num_blocks, in_planes, out_planes, depth):
        hg = []
        # inverse out_planes
        out_planes = out_planes[::-1]
        for i in range(depth):
            # start from the smallest features
            hg.append(self._make_residual(num_blocks, in_planes, out_planes[i + 1]))
            in_planes = out_planes[i + 1]
        return nn.ModuleList(hg)


    def _make_decoder_to_decoder_residual(self, num_blocks, in_planes, out_planes, depth):
        # if we have convs from decoder to decoder
        layers = []
        # inverse out_planes
        out_planes = out_planes[::-1]
        for i in range(depth):
            # start from the smallest features
            layers.append(self._make_residual(num_blocks, out_planes[i+1], out_planes[i+1]))
        return nn.ModuleList(layers)

    def forward(self, x, res, extra=None):
        outputs = []
        for depth in range(self.depth):
            x = self.hg[depth](x)
            if depth == self.stop_upsample_idx:
                x = x + F.max_pool2d(res[depth], kernel_size=3, stride=self.output_stride, padding=1)
                if extra is not None:
                    x = x + F.max_pool2d(extra[depth], kernel_size=3, stride=self.output_stride, padding=1)
            else:
                x = self.upsample(x)
                x = x + res[depth]
                if extra is not None:
                    x = x + extra[depth]
            if self.intermediate_output:
                outputs.append(self.bridge[depth](x))
        if self.intermediate_output:
            return x, outputs
        # if self.output_stride > 1:
        #     x = F.max_pool2d(x, kernel_size=3, stride=self.output_stride, padding=1)
        return x, None


class TriadNet(Backbone):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, cfg, input_shape):
        super(TriadNet, self).__init__()
        # Parameters: num_feats=256, num_stacks=8, num_blocks=1, num_classes=16
        num_branches = cfg.MODEL.TRIAD.NUM_BRANCHES
        num_feats = cfg.MODEL.TRIAD.NUM_FEATURES
        # this now is a list,
        encoder_num_blocks = cfg.MODEL.TRIAD.ENCODER_BLOCKS
        decoder_num_blocks = cfg.MODEL.TRIAD.DECODER_BLOCKS
        self.raf_stride_ratio = cfg.INPUT.RAF_STRIDE_RATIO
        if len(decoder_num_blocks) != num_branches:
            decoder_num_blocks = decoder_num_blocks * num_branches
        self.norm = cfg.MODEL.TRIAD.NORM
        self.expansion = 2
        self.depth = 4
        # Parameters: self.inplanes=256/4=64
        self.stem_in_channels = num_feats // 4
        # Parameters: self.num_feats=256/2=128
        self.stem_out_channels = num_feats // 2
        self.encoder_in_channels = num_feats # 256
        # output channels for each depth, instead of using constant 256
        # [256, 512, 1024, 2048, 2048]
        # self.out_channels = [num_feats * (2 ** i) for i in range(self.depth)] + [num_feats * (2 ** (self.depth - 1))]
        # if above No. params is too large
        self.out_channels = [256, 384, 512, 640, 1024]
        # Parameters: self.num_stacks=8
        self.num_branches = num_branches
        # all levels
        self._out_features = ["decoder_{}".format(i) for i in range(num_branches)]

        # based on the out features from cfg
        # self._out_indice = [self._out_features.index(out_feat) for out_feat in out_features]
        self._out_features = cfg.MODEL.TRIAD.OUT_FEATURES
        self._out_feature_channels = {k: num_feats for k in self._out_features}
        self._out_feature_strides = {k: 2 ** 2 for _, k in enumerate(self._out_features)}
        stem_layers = []
        # stem part 7x7
        stem_layers.append(Conv2d(3, self.stem_in_channels,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=not self.norm,
                          activation=F.relu,
                          norm=get_norm(self.norm, self.stem_in_channels)))
        # Parameters: planes=64, blocks=1, stride=1, output channels = 128
        stem_layers += self._make_residual(self.stem_in_channels, self.stem_out_channels, 1)
        # strided conv in replacement of maxpool
        stem_layers.append(Conv2d(self.stem_out_channels, self.stem_out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=not self.norm,
                              activation=F.relu,
                              norm=get_norm(self.norm, self.stem_out_channels)))
        stem_layers += self._make_residual(self.stem_out_channels, self.stem_out_channels, 1)
        stem_layers += self._make_residual(self.stem_out_channels, self.encoder_in_channels, 1)
        self.stem = nn.Sequential(*stem_layers)
        # Build triad
        self.encoder = TriadEncoder(encoder_num_blocks, self.encoder_in_channels,
                                    self.out_channels, self.depth, norm=self.norm)

        self.decoders = nn.ModuleList([
            TriadDecoder(decoder_num_blocks[0], self.out_channels[-1],
                         self.out_channels, self.depth, norm=self.norm, intermediate_output=True),
            TriadDecoder(decoder_num_blocks[1], self.out_channels[-1],
                         self.out_channels, self.depth, norm=self.norm,
                         output_stride=self.raf_stride_ratio)
        ])
        # self.obj_decoder = TriadDecoder(decoder_num_blocks[0], self.out_channels[-1],
        #                                 self.out_channels, self.depth, norm=self.norm)
        # self.rel_decoder = TriadDecoder(decoder_num_blocks[1], self.out_channels[-1],
        #                                 self.out_channels, self.depth, norm=self.norm,
        #                                 output_stride=self.raf_stride_ratio)


    def _make_residual(self, in_planes, out_planes, num_blocks):
        layers = []
        # increase (double) the number of channels here
        layers.append(BottleneckBlock(in_planes, out_planes,
                                      bottleneck_channels=in_planes))
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_planes, out_planes,
                                          bottleneck_channels=out_planes // self.expansion))

        return layers

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        # res will be a list of lists,
        # dim 0: first residual for first branch, second ...
        # dim 1: from highlevel features to low level
        x, res_all_branches = self.encoder(x)
        # list of list of tensors,
        # res_all_branches[0] for center branch, [1] for raf branch
        res_all_branches = list(zip(*res_all_branches))
        obj_features, shortcuts = self.decoders[0](x, res_all_branches[0])

        outputs.append(obj_features)
        rel_features, _ = self.decoders[1](x, res_all_branches[1], shortcuts)
        outputs.append(rel_features)
        # for decoder, res in zip(self.decoders, res_all_branches):
        #     outputs.append(decoder(x, res))

        return dict(zip(self._out_features, outputs))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



@BACKBONE_REGISTRY.register()
def build_triad_net(cfg, input_shape: ShapeSpec):
    model = TriadNet(cfg, input_shape)
    return model
