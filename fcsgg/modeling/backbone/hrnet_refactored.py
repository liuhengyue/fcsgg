"""
Implementations of HRNet using Detectron2 conv2d wrapper.
The problem of `hrnet.py` is that if we freeze the backbone, the features values will become all zero
at certain point. By refactoring, the problem will be resolved.
Since all the pre-trained weights are trained using the model in `hrnet.py` rather than this file,
this is reserved for future retraining.
Modified from https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation/blob/master/lib/models/pose_higher_hrnet.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Bin Xiao (leoxiaobin@gmail.com)", "Bowen Cheng (bcheng9@illinois.edu)"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from detectron2.layers import ShapeSpec, FrozenBatchNorm2d, Conv2d
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone

from fcsgg.layers import get_norm

logger = logging.getLogger("detectron2")

__all__ = ["build_hrnet_backbone"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm="SyncBN", bn_momentum=0.01):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, stride=stride,
                            kernel_size=3, padding=1, bias=False,
                            activation=torch.relu_,
                            norm=get_norm(norm, planes, momentum=bn_momentum))
        self.conv2 = Conv2d(planes, planes, stride=stride,
                            kernel_size=3, padding=1, bias=False,
                            activation=None,
                            norm=get_norm(norm, planes, momentum=bn_momentum))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = torch.relu_(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm="SyncBN", bn_momentum=0.01):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False,
                            activation=torch.relu_,
                            norm=get_norm(norm, planes, momentum=bn_momentum)
                            )
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                            activation=torch.relu_,
                            norm=get_norm(norm, planes, momentum=bn_momentum)
                            )
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False,
                            activation=None,
                            norm=get_norm(norm, planes * self.expansion, momentum=bn_momentum)
                            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = torch.relu_(out)

        return out


class HighResolutionModule(nn.Module):
    """ HighResolutionModule
    Building block of the PoseHigherResolutionNet (see lower)
    arXiv: https://arxiv.org/abs/1908.10357
    Args:
        num_branches (int): number of branches of the modyle
        blocks (str): type of block of the module
        num_blocks (int): number of blocks of the module
        num_inchannels (int): number of input channels of the module
        num_channels (list): number of channels of each branch
        multi_scale_output (bool): only used by the last module of PoseHigherResolutionNet
    """

    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            multi_scale_output=True,
            norm="SyncBN",
            upsample_mode="nearest",
            bn_momentum=0.01
    ):
        super(HighResolutionModule, self).__init__()
        self.norm = norm
        self.bn_momentum = bn_momentum
        self.upsample_mode = {"mode": upsample_mode, "align_corners": None} if upsample_mode == "nearest" \
            else {"mode": upsample_mode, "align_corners": True}
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = Conv2d(
                self.num_inchannels[branch_index],
                num_channels[branch_index] * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(self.norm, num_channels[branch_index] * block.expansion,
                              momentum=self.bn_momentum))

        layers = []
        layers.append(
            block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample, norm=self.norm)
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm=self.norm))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

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
                        Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False,
                               norm=get_norm(self.norm, num_inchannels[i],
                                             momentum=self.bn_momentum))
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(Conv2d(
                                num_inchannels[j],
                                num_outchannels_conv3x3,
                                3,
                                2,
                                1,
                                bias=False,
                                norm=get_norm(self.norm,
                                              num_outchannels_conv3x3,
                                              momentum=self.bn_momentum)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(Conv2d(
                                num_inchannels[j],
                                num_outchannels_conv3x3,
                                3,
                                2,
                                1,
                                bias=False,
                                activation=torch.relu,
                                norm=get_norm(self.norm,
                                              num_outchannels_conv3x3,
                                              momentum=self.bn_momentum)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode=self.upsample_mode["mode"],
                        align_corners=self.upsample_mode["align_corners"])
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(torch.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class PoseHigherResolutionNet(Backbone):
    """ PoseHigherResolutionNet
    Composed of several HighResolutionModule tied together with ConvNets
    Adapted from the GitHub version to fit with HRFPN and the Detectron2 infrastructure
    arXiv: https://arxiv.org/abs/1908.10357
    """

    def __init__(self, cfg):
        self.weights = cfg.MODEL.HRNET.WEIGHTS
        self.inplanes = cfg.MODEL.HRNET.STEM_INPLANES
        self.norm = cfg.MODEL.HRNET.NORM
        self.bn_momentum = cfg.MODEL.HRNET.BN_MOMENTUM
        self.final_stage_multiscale = cfg.MODEL.HRNET.FINAL_STAGE_MULTISCALE
        self._out_features = cfg.MODEL.HRNET.OUT_FEATURES
        self.upsample_mode = cfg.MODEL.HRNET.UPSAMPLE_MODE
        assert (len(self._out_features) > 1) == self.final_stage_multiscale, "output mismatch."
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False,
                            activation=torch.relu_,
                            norm=get_norm(self.norm, 64, momentum=self.bn_momentum))
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False,
                            activation=torch.relu_,
                            norm=get_norm(self.norm, 64, momentum=self.bn_momentum))
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg.MODEL.HRNET.STAGE2
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg.MODEL.HRNET.STAGE3
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg.MODEL.HRNET.STAGE4
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=self.final_stage_multiscale
        )

        self._out_feature_channels = {}
        self._out_feature_strides = {}

        for i in range(cfg.MODEL.HRNET.STAGE4.NUM_BRANCHES):
            name = "hr" + str(i + 2)
            if name in self._out_features:
                self._out_feature_channels.update(
                    {self._out_features[i]: cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS[-i - 1]}
                )
                self._out_feature_strides.update({self._out_features[i]: 2 ** (i + 2)})

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        Conv2d(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3,
                            1,
                            1,
                            bias=False,
                            activation=torch.relu_,
                            norm=get_norm(self.norm,
                                          num_channels_cur_layer[i],
                                          momentum=self.bn_momentum))
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    )
                    conv3x3s.append(
                        Conv2d(inchannels, outchannels, 3, 2, 1, bias=False,
                               activation=torch.relu_,
                               norm=get_norm(self.norm, outchannels, momentum=self.bn_momentum)
                               )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(self.norm, planes * block.expansion, momentum=self.bn_momentum))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm=self.norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=self.norm))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output,
                    norm=self.norm,
                    upsample_mode=self.upsample_mode
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        assert len(self._out_features) == len(y_list)
        outputs = dict(zip(self._out_features, y_list))
        # outputs = {k: v for k, v in outputs.items() if k in self._out_features}
        return outputs

    def load_pretrained_model(self):
        import re
        def get_match_name(module_name):
            maps = {
                "transition1.1.0.norm.weight"
            }
            module_name = module_name.replace("backbone.", "")
            ind = module_name.find("bn")
            if ind != -1:
                module_ind = module_name[ind + 2]
                module_name = module_name.replace("bn" + module_ind, "conv" + module_ind + ".norm")

            module_name = re.sub(r"\.0\.weight", ".weight", module_name)
            module_name = re.sub(r"\.1\.weight", ".norm.weight", module_name)
            return module_name

        if self.weights:
            try:
                if "//" in self.weights:
                    model_weights = model_zoo.load_url(self.weights)
                else:
                    model_weights = torch.load(self.weights, map_location="cpu")
                    if "model" in model_weights:
                        model_weights = model_weights["model"]
                own_state = self.state_dict()
                pretrained_dict = {}
                for k, v in model_weights.items():
                    # pretrained hrnet
                    self_k = k.replace("module.backbone.body.", "")
                    # old version hrnet
                    # self_k = get_match_name(k)
                    if self_k in own_state:
                        pretrained_dict[self_k] = v
                self.load_state_dict(pretrained_dict)
                logger.info("Loaded weights from {}".format(self.weights))
            except RuntimeError:
                pass
        return self


@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg, input_shape: ShapeSpec):
    model = PoseHigherResolutionNet(cfg).load_pretrained_model()

    return model