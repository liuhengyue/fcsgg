"""
Implementations of Hourglass Network.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Hanbin Dai (daihanbin.ac@gmail.com)", "Feng Zhang (zhangfengwcy@gmail.com)"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, Conv2d, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.resnet import BottleneckBlock

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm="SyncBN"):
        # inplanes -> 2*planes
        super(Bottleneck, self).__init__()

        self.bn1 = get_norm(norm, inplanes)
        # 1x1 conv with stride = 1, inplanes -> planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = get_norm(norm, planes)
        # 3x3 conv with stride = ?, planes -> planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = get_norm(norm, planes)
        # 1x1 conv with stride = 1, planes -> 2*planes
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class HourglassEncoder(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, expansion=2, norm="BN"):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.norm = norm
        self.expansion = expansion
        self.hg = self._make_hour_glass(num_blocks, planes, depth)

    def _make_residual(self, num_blocks, planes):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(self.block(planes * self.expansion,
                                planes,
                                norm=self.norm))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, num_blocks, planes, depth):
        """

        |-------- c1_0 -------- |
        |                       |
        |                       |
        |                       |
        c1 -> c2 ... c2_up ---> + c1_out


        """
        hg = []
        for i in range(depth):
            res = []
            # first block: the residual block from c1 to c1_0
            # second block: c1 after maxpool to c2
            # thrid block: conv blocks before upsample c2 to c2_up
            for _ in range(2):
                res.append(self._make_residual(num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        res = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        low1 = self.hg[n-1][1](low1)

        # if n > 1:
        #     low2 = self._hour_glass_forward(n-1, low1)
        # else:
        if n == 1:
            # the smallest feature maps does not need skip connections
            low1 = self.hg[n-1][2](low1)
        return res, low1

    def forward(self, x):
        res_features = []
        for depth in reversed(range(1, self.depth)):
            res, x = self._hour_glass_forward(depth, x)
            res_features.append(res)
        # make resfeatures from small to large
        res_features = res_features[::-1]
        return x, res_features

class HourglassDecoder(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, expansion=2, norm="BN"):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.norm = norm
        self.expansion = expansion
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(num_blocks, planes, depth)

    def _make_residual(self, num_blocks, planes):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(self.block(planes * self.expansion,
                                planes,
                                norm=self.norm))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            hg.append(self._make_residual(num_blocks, planes))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x, res):
        for depth in range(1, self.depth):
            x = self.hg[depth - 1](x)
            x = self.upsample(x)
            x = x + res[depth - 1]
        return x

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, expansion=2, norm="BN"):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.norm = norm
        self.expansion = expansion
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(num_blocks, planes, depth)

    def _make_residual(self, num_blocks, planes):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(self.block(planes * self.expansion,
                                planes,
                                norm=self.norm))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(self._make_residual(num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


# class Hourglass(nn.Module):
#     def __init__(self, block, num_blocks, planes, depth, expansion=2, norm="BN"):
#         super(Hourglass, self).__init__()
#         self.depth = depth
#         self.block = block
#         self.norm = norm
#         self.expansion = expansion
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.hg = self._make_hour_glass(num_blocks, planes, depth)
#
#     def _make_residual(self, num_blocks, planes):
#         layers = []
#         for _ in range(0, num_blocks):
#             layers.append(self.block(planes * self.expansion,
#                                 planes,
#                                 norm=self.norm))
#         return nn.Sequential(*layers)
#
#     def _make_hour_glass(self, num_blocks, planes, depth):
#         hg = []
#         for i in range(depth):
#             res = []
#             for _ in range(3):
#                 res.append(self._make_residual(num_blocks, planes))
#             if i == 0:
#                 res.append(self._make_residual(num_blocks, planes))
#             hg.append(nn.ModuleList(res))
#         return nn.ModuleList(hg)
#
#     def _hour_glass_forward(self, n, x):
#         up1 = self.hg[n-1][0](x)
#         low1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#         low1 = self.hg[n-1][1](low1)
#
#         if n > 1:
#             low2 = self._hour_glass_forward(n-1, low1)
#         else:
#             low2 = self.hg[n-1][3](low1)
#         low3 = self.hg[n-1][2](low2)
#         up2 = self.upsample(low3)
#         out = up1 + up2
#         return out
#
#     def forward(self, x):
#         return self._hour_glass_forward(self.depth, x)



class HourglassNet(Backbone):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, cfg, input_shape):
        super(HourglassNet, self).__init__()
        # Parameters: num_feats=256, num_stacks=8, num_blocks=1, num_classes=16
        block = Bottleneck
        num_stacks = cfg.MODEL.HOURGLASS.NUM_STACKS
        num_feats = cfg.MODEL.HOURGLASS.NUM_FEATURES
        # this now is a list,
        num_blocks = cfg.MODEL.HOURGLASS.NUM_BLOCKS
        if len(num_blocks) != num_stacks:
            num_blocks = num_blocks * num_stacks
        self.norm = cfg.MODEL.HOURGLASS.NORM
        self.expansion = 2
        out_features = cfg.MODEL.HOURGLASS.OUT_FEATURES

        # Parameters: self.inplanes=256/4=64
        self.inplanes = num_feats // 4
        # Parameters: self.num_feats=256/2=128
        self.num_feats = num_feats // 2
        # Parameters: self.num_stacks=8
        self.num_stacks = num_stacks
        # all levels
        self._out_features = ["hg_stack_{}".format(i) for i in range(num_stacks)]

        # based on the out features from cfg
        # self._out_indice = [self._out_features.index(out_feat) for out_feat in out_features]
        self._out_features = out_features
        self._out_feature_channels = {k: num_feats for k in self._out_features}
        self._out_feature_strides = {k: 2 ** 2 for _, k in enumerate(self._out_features)}

        # keep the hg design
        self.conv1 = Conv2d(3, self.inplanes,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=not self.norm,
                          activation=F.relu,
                          norm=get_norm(self.norm, self.inplanes))
        # Parameters: planes=64, blocks=1, stride=1, output channels = 128
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        # this helps preventing feature size flooring
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build hourglass modules
        # ch = self.num_feats * self.expansion
        hg_stages, shortcut_stages, fc, score, fc_, score_ = [], [], [], [], [], []
        self.stage_names = []
        for i in range(num_stacks):
            stage_name = "hg_stack_" + str(i)
            # Parameters: num_blocks=4, self.num_feats=128
            hg = Hourglass(block, num_blocks[i], self.num_feats, 4, norm=self.norm)
            shortcut = self._make_residual(block, self.num_feats, num_blocks[i])
            hg_stages.append(hg)
            shortcut_stages.append(shortcut)
            self.stage_names.append(stage_name)

            # fc.append(self._make_fc(ch, ch))
            # score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            # if i < num_stacks-1:
            #     fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
            #     score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg_stages)
        self.res = nn.ModuleList(shortcut_stages)
        # self.fc = nn.ModuleList(fc)
        # self.score = nn.ModuleList(score)
        # self.fc_ = nn.ModuleList(fc_)
        # self.score_ = nn.ModuleList(score_)


    def _make_residual(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = Conv2d(self.inplanes, planes * self.expansion,
                               kernel_size=1,
                               stride=stride,
                               bias=not self.norm,
                               activation=None,
                               norm=get_norm(self.norm, planes * self.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = {}
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # output a single/numbers of stack output
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            if i < self.num_stacks - 1:
                x = x + y
            if self.stage_names[i] in self._out_features:
                out[self.stage_names[i]] = y


        return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

class DualDecoderHourglass(HourglassNet):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, cfg, input_shape):
        assert cfg.MODEL.HOURGLASS.NUM_STACKS == 3
        super(DualDecoderHourglass, self).__init__(cfg, input_shape)

    def forward(self, x):
        out = {}
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # output a single/numbers of stack output
        # base hourglass encoder
        y = self.hg[0](x)
        y = self.res[0](y)
        x = x + y
        # center decoder and raf decoder
        for i in range(1, self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            if self.stage_names[i] in self._out_features:
                out[self.stage_names[i]] = y

        return out



@BACKBONE_REGISTRY.register()
def build_hourglass_net(cfg, input_shape: ShapeSpec):
    model = HourglassNet(cfg, input_shape)
    return model

@BACKBONE_REGISTRY.register()
def build_dual_decoder_hourglass_net(cfg, input_shape: ShapeSpec):
    model = DualDecoderHourglass(cfg, input_shape)
    return model