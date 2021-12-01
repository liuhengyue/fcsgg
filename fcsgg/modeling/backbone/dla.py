"""
Implementations of Deep Layer Aggregation (DLA).
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Shihua Liang https://github.com/sxhxliang/detectron2_backbone"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import FPN
from detectron2.layers import ShapeSpec, Conv2d, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7



def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class LastLevelP6(nn.Module):
    """
    Modified from the moduel in FPN
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]

class BasicBlock(nn.Module):
    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(inplanes, planes,
                            kernel_size=3,
                            stride=stride, padding=dilation,
                            bias=False, dilation=dilation,
                            norm=get_norm(cfg.MODEL.DLA.NORM, planes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes,
                            kernel_size=3,
                            stride=1, padding=dilation,
                            bias=False, dilation=dilation,
                            norm=get_norm(cfg.MODEL.DLA.NORM, planes))

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=1,
                           bias=False, padding=(kernel_size - 1) // 2,
                           norm=get_norm(cfg.MODEL.DLA.NORM, out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, cfg, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, downsample_type="conv"):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(cfg, in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(cfg, out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(cfg, levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(cfg, levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(cfg, root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            if downsample_type == "conv":
                self.downsample = Conv2d(in_channels, in_channels,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         bias=False,
                                         norm=get_norm(cfg.MODEL.DLA.NORM, in_channels),
                                         activation=F.relu)
            else:
                self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=1, bias=False,
                                  norm=get_norm(cfg.MODEL.DLA.NORM, out_channels))


    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(Backbone):
    def __init__(self, cfg, levels, channels, block=BasicBlock, residual_root=False):
        super(DLA, self).__init__()
        self.cfg = cfg
        self.freeze = cfg.MODEL.DLA.FREEZE
        self.weights = cfg.MODEL.DLA.WEIGHTS
        self.channels = channels

        self._out_features = ["level{}".format(i) for i in range(6)]
        self._out_feature_channels = {k: channels[i] for i, k in enumerate(self._out_features)}
        self._out_feature_strides = {k: 2 ** i for i, k in enumerate(self._out_features)}

        self.base_layer = Conv2d(3, channels[0],
                                 kernel_size=7, stride=1,
                                 padding=3, bias=False,
                                 activation=F.relu,
                                 norm=get_norm(cfg.MODEL.DLA.NORM, channels[0]))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(cfg, levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(cfg, levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(cfg, levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(cfg, levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # self.avgpool = nn.AvgPool2d(pool_size)
        # self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
        #                     stride=1, padding=0, bias=True)

        # init weights here
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        if self.freeze:
            self.freeze_model()
        if self.weights:
            self.load_pretrained_model()



    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                Conv2d(inplanes, planes,
                       kernel_size=1, stride=1, bias=False,
                       norm=get_norm(self.cfg.MODEL.DLA.NORM, planes)),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(Conv2d(inplanes, planes,
                       kernel_size=3,
                       stride=stride if i == 0 else 1,
                       padding=dilation, bias=False, dilation=dilation,
                       norm=get_norm(self.cfg.MODEL.DLA.NORM, planes),
                       activation=F.relu))
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = {}
        x = self.base_layer(x)
        for i in range(6):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            y[name] = x
        return y

    def freeze_model(self):
        # freeze the whole DLA model
        for param in self.parameters():
            param.requires_grad = False

    def load_pretrained_model(self):
        model_weights = model_zoo.load_url(self.weights)
        own_state = self.state_dict()
        pretrained_dict = {k: v for k, v in model_weights.items() if k in own_state}
        self.load_state_dict(pretrained_dict)# DLA with upsample, modules


def dla34(cfg):  # DLA-34
    model = DLA(cfg, [1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock)
    return model


def dla46_c(cfg, pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA(cfg, [1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46_c')
    return model


def dla46x_c(cfg, pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA(cfg, [1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46x_c')
    return model


def dla60x_c(cfg, pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA(cfg, [1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x_c')
    return model


def dla60(cfg):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA(cfg, [1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck)
    return model


def dla60x(cfg, pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA(cfg, [1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x')
    return model


def dla102(cfg, pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA(cfg, [1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102')
    return model


def dla102x(cfg, pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA(cfg, [1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x')
    return model


def dla102x2(cfg, pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA(cfg, [1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x2')
    return model


def dla169(cfg, pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA(cfg, [1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla169')
    return model

@BACKBONE_REGISTRY.register()
def build_dla_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, "Freezing layers does not be supported for DLA"

    depth_to_creator = {"DLA34": dla34, "DLA60": dla60}
    return depth_to_creator[cfg.MODEL.DLA.CONV_BODY](cfg)


@BACKBONE_REGISTRY.register()
def build_dla_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, "Freezing layers does not be supported for DLA"

    depth_to_creator = {"DLA34": dla34}
    bottom_up = depth_to_creator[cfg.MODEL.DLA.CONV_BODY](cfg)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_fcos_dla_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, "Freezing layers does not be supported for DLA"

    depth_to_creator = {"DLA34": dla34}
    bottom_up = depth_to_creator[cfg.MODEL.DLA.CONV_BODY](cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels

    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    elif top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    else:
        raise NotImplementedError()

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone