"""
Core implementations of detection heads in a single feature scale.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = [""]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import math
import inspect
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.layers import cat, Conv2d, CNNBlockBase
from detectron2.layers.batch_norm import FrozenBatchNorm2d
import fvcore.nn.weight_init as weight_init
from fcsgg.structures import SceneGraph

import fcsgg.utils.centernet_utils as centernet_utils
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from fcsgg.layers import get_norm, CoordConv
from fcsgg.layers import MultiNormConvBlock, MultiNormDeformBlock, ConvLSTM, DA_Module
from fcsgg.modeling.losses import RAFLoss, RelationLoss
HEADS_REGISTRY = Registry("HEADS")
HEADS_REGISTRY.__doc__ = """
Registry for heads in a single-stage model.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Head`.
"""

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.01

class GeneralHeads(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    @configurable
    def __init__(self, *, num_classes, in_features, in_channels,
                 cls_bias_value, ct_loss_weight, wh_loss_weight, reg_loss_weight,
                 hm_loss_func, freeze_heads, use_gt_box, use_gt_object_label,
                 output_strides):
        super().__init__()
        # fmt: off
        self.num_classes     = num_classes
        self.in_features     = in_features
        self.in_channels     = in_channels
        self.cls_bias_value  = cls_bias_value
        self.ct_loss_weight  = ct_loss_weight
        self.wh_loss_weight  = wh_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.freeze_heads    = freeze_heads
        self.hm_loss_func    = getattr(centernet_utils, hm_loss_func)
        self.use_gt_box      = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.output_strides  = output_strides
        self.single_scale = len(self.output_strides) == 1
        # fmt: on


    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        The forward function will return a dict of loss {name: loss_val}
        """
        raise NotImplementedError()


    def loss(self, predictions, targets, after_activation=True) -> Dict[str, torch.Tensor]:
        """
        The basic loss of centerNet heads, namely center heatmaps, size heatmaps, and offsets heatmaps.
        """
        pred_score = predictions['cls']
        if not after_activation:
            pred_score = torch.sigmoid(pred_score)

        num_instances = [len(x.gt_wh) for x in targets]

        # multi-use index, dim0 - image index in B, dim1 - class index in center maps, dim2 - spatial loc of center
        index = [torch.stack((torch.ones(num_instances[i], dtype=torch.long, device=x.gt_index.device) * i,
                              # x.gt_classes,
                              x.gt_index))
                 for i, x in enumerate(targets)]

        index = cat(index, dim=1)

        loss_cls = self.hm_loss_func(pred_score, targets)

        gt_wh = cat([x.gt_wh for x in targets], dim=0)
        gt_reg = cat([x.gt_reg for x in targets], dim=0)

        # if regression target at the same location, choose a random object
        # filtered_index, ori_inds = torch.unique(index, dim=1, return_inverse=True)
        # if filtered_index.numel() != 0:
        #     gt_wh = gt_wh[ori_inds[:filtered_index.size(1)]]
        #     gt_reg = gt_reg[ori_inds[:filtered_index.size(1)]]
        # index = filtered_index
        # width and height loss, better version
        loss_wh = centernet_utils.reg_l1_loss(predictions['wh'], index, gt_wh)

        # regression loss
        loss_reg = centernet_utils.reg_l1_loss(predictions['reg'], index, gt_reg)

        loss_cls *= self.ct_loss_weight
        loss_wh  *= self.wh_loss_weight
        loss_reg *= self.reg_loss_weight

        loss = {
            "loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
        }
        # print(loss)
        return loss

    def freeze(self):
        if len(self.freeze_heads) > 0:
            for head in self.freeze_heads:
                if hasattr(self, head):
                    getattr(self, head).freeze()
        return self

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features = cfg.MODEL.HEADS.IN_FEATURES
        in_channels = [input_shape[k].channels for k in in_features]
        return {
            "num_classes": cfg.MODEL.HEADS.NUM_CLASSES,
            "in_features": in_features,
            "in_channels": in_channels,
            "cls_bias_value": cfg.MODEL.HEADS.CLS_BIAS_VALUE,
            "ct_loss_weight": cfg.MODEL.HEADS.LOSS.CT_WEIGHT,
            "wh_loss_weight": cfg.MODEL.HEADS.LOSS.WH_WEIGHT,
            "reg_loss_weight": cfg.MODEL.HEADS.LOSS.REG_WEIGHT,
            "freeze_heads": cfg.MODEL.HEADS.FREEZE,
            "hm_loss_func": cfg.MODEL.HEADS.LOSS.HEATMAP_LOSS_TYPE,
            "use_gt_box": cfg.RELATION.USE_GT_BOX,
            "use_gt_object_label": cfg.RELATION.USE_GT_OBJECT_LABEL,
            "output_strides": cfg.MODEL.HEADS.OUTPUT_STRIDES
        }

class SingleHead(CNNBlockBase):

    def __init__(self, in_channel, out_channel, stride,
                 conv_dims, dilation=1, bias_fill=False, bias_value=0,
                 conv_norm="", kernel_size=3,
                 deformable_on=False,
                 deformable_first=True,
                 bottleneck_channels=64,
                 activation=None,
                 down_ratio=1,
                 up_sample=True,
                 split_pred=False,
                 add_coord=False):
        super(SingleHead, self).__init__(in_channel, out_channel, stride)
        self.activation = activation
        self.conv_norm_relus = []
        self.down_ratio = down_ratio
        self.up_sample = up_sample
        self.split_pred = split_pred
        cur_channels = in_channel
        cur_channels = cur_channels + 2 if add_coord else cur_channels
        deformable_idx = 0 if deformable_first else len(conv_dims) - 1
        for k, conv_dim in enumerate(conv_dims):
            stride = 2 if self.down_ratio > 1 and k < math.log2(self.down_ratio) else 1
            # if deformable_on and k == deformable_idx:
            if deformable_on:
                conv = DeformBottleneckBlock(cur_channels, conv_dim,
                                             bottleneck_channels=conv_dim//2,
                                             stride=stride,
                                             # stride_in_1x1 = stride == 1,
                                             norm=conv_norm,
                                             deform_modulated=True,
                                             deform_num_groups=1,
                                             dilation=1
            )
            # elif add_coord and k == 0:
            #     conv = CoordConv(cur_channels, conv_dim,
            #                   kernel_size=kernel_size,
            #                   stride=stride,
            #                   padding=(kernel_size * dilation - 1) // 2,
            #                   dilation=dilation,
            #                   bias=not conv_norm,
            #                   activation=F.relu,
            #                   norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM))
            else:
                conv = Conv2d(cur_channels, conv_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size * dilation - 1) // 2,
                              dilation=dilation,
                              bias=not conv_norm,
                              activation=F.relu,
                              norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM))
            self.add_module("head_fcn{}".format(k), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        if self.split_pred:
            self.out_conv_x = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            self.out_conv_y = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            nn.init.xavier_normal_(self.out_conv_x.weight)
            self.out_conv_x.bias.data.fill_(bias_value)
            nn.init.xavier_normal_(self.out_conv_y.weight)
            self.out_conv_y.bias.data.fill_(bias_value)
        else:
            self.out_conv = Conv2d(cur_channels, out_channel, kernel_size=1)
            # initialization for output layer
            nn.init.xavier_normal_(self.out_conv.weight)
            self.out_conv.bias.data.fill_(bias_value)
        if not deformable_on:
            for layer in self.conv_norm_relus:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if self.down_ratio > 1 and self.up_sample:
            x = F.interpolate(x,
                             scale_factor=self.down_ratio,
                             mode="nearest",
                             align_corners=None)
        if self.split_pred:
            x_0 = self.out_conv_x(x)
            x_1 = self.out_conv_y(x)
            x = torch.stack((x_0, x_1), dim=2)
        else:
            x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MultiBNSingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, stride,
                 conv_dims, dilation=1, bias_fill=False, bias_value=0,
                 conv_norm="", kernel_size=3,
                 deformable_on=False,
                 deformable_first=True,
                 bottleneck_channels=64,
                 activation=None,
                 output_strides=[4, 8, 16, 32],
                 down_ratio=1,
                 bn_momentum=0.01,
                 up_ratio=1,
                 split_pred=False,
                 output_feat=False,
                 add_relationess=False):
        super(MultiBNSingleHead, self).__init__()
        self.activation = activation
        self.conv_norm = conv_norm
        self.conv_norm_relus = []
        self.down_ratio = down_ratio
        self.up_ratio = up_ratio
        self.split_pred = split_pred
        self.output_feat = output_feat
        self.add_relationess = add_relationess
        cur_channels = in_channel
        deformable_idx = 0 if deformable_first else len(conv_dims) - 1
        for k, conv_dim in enumerate(conv_dims):
            stride = 2 if self.down_ratio > 1 and k < math.log2(self.down_ratio) else 1
            # norm and activation are included in the wrapper
            # if deformable_on and k == deformable_idx:
            if deformable_on:
                conv = MultiNormDeformBlock(cur_channels, conv_dim,
                                             output_strides,
                                             bottleneck_channels=conv_dim//2,
                                             norm=conv_norm,
                                             deform_modulated=False,
                                             deform_num_groups=1,
                                             dilation=1)

            else:

                conv = MultiNormConvBlock(cur_channels, conv_dim,
                                          output_strides,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=(kernel_size * dilation - 1) // 2,
                                          dilation=dilation,
                                          norm=conv_norm,
                                          bn_momentum=bn_momentum,
                                          activation=F.relu)

            self.add_module("head_fcn{}".format(k), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim
        if self.split_pred:
            self.out_conv_x = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            self.out_conv_y = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            nn.init.xavier_normal_(self.out_conv_x.weight)
            self.out_conv_x.bias.data.fill_(bias_value)
            nn.init.xavier_normal_(self.out_conv_y.weight)
            self.out_conv_y.bias.data.fill_(bias_value)
        else:
            self.out_conv = Conv2d(cur_channels, out_channel, kernel_size=1)
            # initialization for output layer
            nn.init.xavier_normal_(self.out_conv.weight)
            self.out_conv.bias.data.fill_(bias_value)
        if not deformable_on:
            for layer in self.conv_norm_relus:
                weight_init.c2_msra_fill(layer)
        if self.add_relationess:
            self.relationess = Conv2d(cur_channels, 1, kernel_size=1, activation=torch.sigmoid_)



    def forward(self, x, stride):
        for layer in self.conv_norm_relus:
            x = layer(x, stride)
        if self.up_ratio > 1:
            x = F.interpolate(x,
                             scale_factor=self.up_ratio,
                             mode="nearest",
                             align_corners=None)
        if self.split_pred:
            x_0 = self.out_conv_x(x)
            x_1 = self.out_conv_y(x)
            if self.add_relationess:
                relation_score = self.relationess(x)
                x_0 = x_0 * relation_score
                x_1 = x_1 * relation_score
            out = torch.stack((x_0, x_1), dim=2)
        else:
            out = self.out_conv(x)
        if self.activation is not None:
            out = self.activation(out)
        if self.output_feat:
            return out, x
        return out

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

@HEADS_REGISTRY.register()
class UnionHead(GeneralHeads):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        norm = cfg.MODEL.HEADS.NORM
        shared = cfg.MODEL.HEADS.SHARED
        conv_dims = [conv_dim] * num_conv
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        in_channels = self.in_channels[0]
        # does not influence anything
        self.output_stride = self.output_strides[0]

        if not shared:
            assert len(self.in_features) == 1 and len(self.in_channels) == 1

        self.output_dims = [self.num_classes, self.num_predicates * 2 if self.relation_on else 0, 2, 2]

        self.head = SingleHead(
            in_channels,
            sum(self.output_dims),
            self.output_stride,
            conv_dims,
            conv_norm=norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=None
        )

        if self.training:
            self.raf_loss_evaluator = RAFLoss(cfg)

    def forward(
            self,
            features: Union[Dict[str, torch.Tensor], torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]

        # e.g. {"p2": tensor1, "p3": tensor2}
        if isinstance(features, Dict):
            features = [features[f] for f in self.in_features][0]

        features = self.head(features)
        cls, rafs, wh, reg = torch.split(features, self.output_dims, dim=1)
        cls = torch.sigmoid_(cls)
        # rafs = torch.tanh_(rafs)
        # wh = torch.relu_(wh)
        # reg = torch.sigmoid_(reg)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.relation_on:
            preds.update({'raf': rafs})
        if self.training:
            losses.update(self.loss(preds, targets))
            if self.relation_on:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds

@HEADS_REGISTRY.register()
class CenternetHeads(GeneralHeads):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        norm = cfg.MODEL.HEADS.NORM
        # we only use shared head
        shared = len(self.in_features) > 1
        conv_dims = [conv_dim] * num_conv
        # does not influence anything
        self.output_stride = self.output_strides[0]
        add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD

        if not shared:
            assert len(self.in_features) == 1 and len(self.in_channels) == 1

        in_channels = self.in_channels[0]
        self.cls_head = SingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=torch.sigmoid_,
            add_coord=add_coord
        )
        self.wh_head = SingleHead(in_channels, 2, self.output_stride,
                       conv_dims, conv_norm=norm,
                                  activation=None,
                                  add_coord=add_coord)
        self.reg_head = SingleHead(in_channels, 2, self.output_stride,
                                   conv_dims, conv_norm=norm,
                                   activation=None,
                                   add_coord=add_coord)

    def forward(
            self,
            features: Union[Dict[str, torch.Tensor], torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]

        # e.g. {"p2": tensor1, "p3": tensor2}
        if isinstance(features, Dict):
            features = [features[f] for f in self.in_features][0]

        cls = self.cls_head(features)
        wh = self.wh_head(features)
        reg = self.reg_head(features)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.training:
            losses.update(self.loss(preds, targets))

        return losses, preds


@HEADS_REGISTRY.register()
class CenternetRelationLSTMHeads(CenternetHeads):
    """
    Extended heads with relation affinity field prediction head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        in_channels = self.in_channels[0]
        if self.relation_on:
            num_steps = 3
            self.raf_head = nn.Sequential(
                Conv2d(in_channels, conv_dim,
                       kernel_size=kernel_size,
                       stride=1,
                       padding=1,
                       dilation=1,
                       bias=not conv_norm,
                       activation=F.relu,
                       norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM)),
                ConvLSTM(input_channels=conv_dim,
                         hidden_channels=[2 * self.num_predicates] * num_steps,
                         kernel_size=3,
                         step=num_steps,
                         effective_step=[0, 1, 2])
                )
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)


    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses, preds = super().forward(features, targets)
        if self.relation_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features][0]
            if self.training and self.single_scale:
                targets = [target[0] for target in targets]
            rafs = self.raf_head(features)[0]
            preds.update({'raf': torch.tanh(rafs[-1])})
            if self.training:
                loss_raf = sum([self.raf_loss_evaluator(torch.tanh(raf), targets)
                                      for raf in rafs])
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})
        return losses, preds

@HEADS_REGISTRY.register()
class CenternetRelationHeads(CenternetHeads):
    """
    Extended heads with relation affinity field prediction head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.raf_type = cfg.INPUT.RAF_TYPE
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        in_channels = self.in_channels[0]
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        raf_conv_dims = [cfg.MODEL.HEADS.RAF.CONV_DIM] * cfg.MODEL.HEADS.RAF.NUM_CONV
        # raf_conv_dims = cfg.MODEL.HEADS.RAF.CONV_DIMS
        split_pred = cfg.MODEL.HEADS.RAF.SPLIT
        add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD
        if self.relation_on:
            raf_activation_func = torch.sigmoid_ if self.raf_type == "point" else torch.tanh_
            self.raf_head = SingleHead(
                                in_channels,
                                2 * self.num_predicates,
                                self.output_stride,
                                raf_conv_dims,
                                kernel_size=kernel_size,
                                conv_norm=conv_norm,
                                bias_fill=True,
                                deformable_on=deformable_on,
                                dilation=self.raf_dilation,
                                bottleneck_channels=bottleneck_channels,
                                activation=raf_activation_func,
                                bias_value=self.cls_bias_value if self.raf_type == "point" else 0,
                                down_ratio=self.down_ratio,
                                split_pred=split_pred,
                                add_coord=add_coord
                            )
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)
                # self.rel_loss_evaluator = RelationLoss()



    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # if self.training:
            # only take one scale
            # targets = [target[0] for target in targets]
        losses, preds = super().forward(features, targets)
        if self.relation_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features][0]
            if self.training and self.single_scale:
                targets = [target[0] for target in targets]
            rafs = self.raf_head(features)
            preds.update({'raf': rafs})
            if self.training:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})
                # loss_rel = self.rel_loss_evaluator(rafs, preds['cls'], targets)
                # losses.update({'loss_rel': loss_rel})
        return losses, preds


@HEADS_REGISTRY.register()
class CenternetRelationAttnHeads(CenternetHeads):
    """
    Extended heads with relation affinity field prediction head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        in_channels = self.in_channels[0]
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        if self.relation_on:
            self.raf_head = SingleHead(
                                in_channels,
                                2 * self.num_predicates,
                                self.output_stride,
                                conv_dims,
                                kernel_size=kernel_size,
                                conv_norm=conv_norm,
                                bias_fill=True,
                                deformable_on=deformable_on,
                                dilation=self.raf_dilation,
                                bottleneck_channels=bottleneck_channels,
                                activation=torch.tanh_,
                                down_ratio=self.down_ratio,
                                up_sample=False
                            )

            self.attn_layer = DA_Module(self.num_classes, self.num_predicates)

            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)



    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # if self.training:
            # only take one scale
            # targets = [target[0] for target in targets]
        losses, preds = super().forward(features, targets)
        if self.relation_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features][0]
            if self.training and self.single_scale:
                targets = [target[0] for target in targets]
            rafs = self.raf_head(features)
            rafs = self.attn_layer(preds["cls"], rafs)
            preds.update({'raf': rafs})
            if self.training:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})
        return losses, preds

@HEADS_REGISTRY.register()
class CenternetCascadeHeads(GeneralHeads):
    """
    The raf head is computed from backbone/neck features plus cls and wh features.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE

        in_channels = self.in_channels[0]
        self.cls_head = SingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=conv_norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=torch.sigmoid
        )
        self.wh_head = SingleHead(in_channels, 2, self.output_stride,
                       conv_dims, conv_norm=conv_norm,
                                  activation=F.relu)
        self.reg_head = SingleHead(in_channels, 2, self.output_stride,
                                   conv_dims, conv_norm=conv_norm,
                                   activation=torch.sigmoid)
        if self.relation_on:
            self.transition = Conv2d(self.num_classes, in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=2,
                                  dilation=2,
                                  bias=True,
                                  activation=torch.sigmoid,
                                  norm=None)
            self.raf_head = SingleHead(in_channels * 2,
                                       2 * self.num_predicates,
                                       self.output_stride,
                                       conv_dims,
                                       kernel_size=kernel_size,
                                       conv_norm=conv_norm,
                                       bias_fill=True,
                                       deformable_on=deformable_on,
                                       dilation=self.raf_dilation,
                                       bottleneck_channels=bottleneck_channels,
                                       activation=torch.tanh
                                       )

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]
        if isinstance(features, Dict):
            features = [features[f] for f in self.in_features][0]

        cls = self.cls_head(features)
        wh = self.wh_head(features)
        reg = self.reg_head(features)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.relation_on:
            if self.training:
                ct_maps_per_scale = torch.stack([target.gt_ct_maps for target in targets], dim=0)
            else:
                ct_maps_per_scale = cls.detach()
            trans_feat = self.transition(ct_maps_per_scale)
            # cat_features = trans_feat * features
            cat_features = torch.cat((features, trans_feat), dim=1)
            rafs = self.raf_head(cat_features)
            preds.update({'raf': rafs})

        if self.training:
            losses.update(self.loss(preds, targets))
            if self.relation_on:
                loss_raf = centernet_utils.raf_loss(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds


@HEADS_REGISTRY.register()
class DualDecoderHeads(GeneralHeads):
    """
    The raf head is computed from backbone/neck features plus cls and wh features.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE

        in_channels = self.in_channels[0]
        self.cls_head = SingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=conv_norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=torch.sigmoid
        )
        self.wh_head = SingleHead(in_channels, 2, self.output_stride,
                                  conv_dims,
                                  bias_fill=True,
                                  conv_norm=conv_norm,
                                  activation=None)
        self.reg_head = SingleHead(in_channels, 2, self.output_stride,
                                   conv_dims,
                                   conv_norm=conv_norm,
                                   activation=None)
        if self.relation_on:
            self.raf_head = SingleHead(in_channels,
                                       2 * self.num_predicates,
                                       self.output_stride,
                                       conv_dims,
                                       kernel_size=kernel_size,
                                       conv_norm=conv_norm,
                                       bias_fill=True,
                                       deformable_on=deformable_on,
                                       dilation=self.raf_dilation,
                                       bottleneck_channels=bottleneck_channels,
                                       activation=None
                                       )

            self.raf_loss_evaluator = RAFLoss(cfg)

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]
        # this should be list of 2 elements, first for center, second for raf
        features = [features[f] for f in self.in_features]

        cls = self.cls_head(features[0])
        wh = self.wh_head(features[0])
        reg = self.reg_head(features[0])
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.relation_on:
            rafs = self.raf_head(features[1])
            preds.update({'raf': rafs})

        if self.training:
            losses.update(self.loss(preds, targets))
            if self.relation_on:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds

@HEADS_REGISTRY.register()
class SwitchNormHeads(GeneralHeads):
    """
    The raf head is computed from backbone/neck features plus cls and wh features.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.relation_only = cfg.RELATION.RELATION_ONLY
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        raf_config = cfg.MODEL.HEADS.RAF
        raf_conv_dims = [raf_config.CONV_DIM] * raf_config.NUM_CONV
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        bn_momentum = cfg.MODEL.HEADS.BN_MOMENTUM
        split_pred = cfg.MODEL.HEADS.RAF.SPLIT
        assert not self.single_scale
        add_relationess = cfg.MODEL.HEADS.RAF.ADD_RELATIONESS

        in_channels = self.in_channels[0]
        if not self.relation_only:
            self.cls_head = MultiBNSingleHead(
                in_channels,
                self.num_classes,
                self.output_stride,
                conv_dims,
                conv_norm=conv_norm,
                bias_fill=True,
                bias_value=self.cls_bias_value,
                activation=torch.sigmoid_,
                output_strides=self.output_strides,
                bn_momentum=bn_momentum
            )
            self.wh_head = MultiBNSingleHead(in_channels, 2, self.output_stride,
                                              conv_dims,
                                              bias_fill=True,
                                              conv_norm=conv_norm,
                                              activation=None,
                                              output_strides=self.output_strides,
                                              bn_momentum=bn_momentum)
            self.reg_head = MultiBNSingleHead(in_channels, 2, self.output_stride,
                                              conv_dims,
                                              conv_norm=conv_norm,
                                              activation=None,
                                              output_strides=self.output_strides,
                                              bn_momentum=bn_momentum)
        if self.relation_on:
            self.raf_head = MultiBNSingleHead(in_channels,
                                       2 * self.num_predicates,
                                       self.output_stride,
                                       raf_conv_dims,
                                       kernel_size=kernel_size,
                                       conv_norm=conv_norm,
                                       bias_fill=True,
                                       deformable_on=deformable_on,
                                       dilation=self.raf_dilation,
                                       bottleneck_channels=bottleneck_channels,
                                       activation=None,
                                       output_strides=self.output_strides,
                                       down_ratio=self.down_ratio,
                                       bn_momentum=bn_momentum,
                                       up_ratio=self.down_ratio,
                                       split_pred=split_pred,
                                       add_relationess=add_relationess
                                       )
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)

    def forward(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses, preds = {}, {}
        if self.training:
            assert targets
        if not self.relation_only:
            # features should be tuple of 2 elements, first for center, second for raf
            cls = self.cls_head(features, stride)
            wh = self.wh_head(features, stride)
            reg = self.reg_head(features, stride)
            preds.update({
                'cls': cls,
                'wh': wh,
                'reg': reg
            })
            if self.training:
                losses.update(self.loss(preds, targets))

        if self.relation_on:
            rafs = self.raf_head(features, stride)
            preds.update({'raf': rafs})
            if self.training:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds

@HEADS_REGISTRY.register()
class DualBranchSwitchNormHeads(SwitchNormHeads):
    """
    The raf head is computed from backbone/neck features plus cls and wh features.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def forward(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses = {}
        if self.training:
            assert targets
        # features should be tuple of 2 elements, first for center, second for raf
        cls = self.cls_head(features[0], stride)
        wh = self.wh_head(features[0], stride)
        reg = self.reg_head(features[0], stride)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.relation_on:
            rafs = self.raf_head(features[1], stride)
            preds.update({'raf': rafs})

        if self.training:
            losses.update(self.loss(preds, targets))
            if self.relation_on:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds

@HEADS_REGISTRY.register()
class DualBranchHeads(GeneralHeads):
    """
    The raf head is computed from backbone/neck features plus cls and wh features.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        assert not self.single_scale

        in_channels = self.in_channels[0]
        self.cls_head = SingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=conv_norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=torch.sigmoid_
        )
        self.wh_head = SingleHead(in_channels, 2, self.output_stride,
                                          conv_dims,
                                          bias_fill=True,
                                          conv_norm=conv_norm,
                                          activation=None)
        self.reg_head = SingleHead(in_channels, 2, self.output_stride,
                                          conv_dims,
                                          conv_norm=conv_norm,
                                          activation=None)
        if self.relation_on:
            self.raf_head = SingleHead(in_channels,
                                       2 * self.num_predicates,
                                       self.output_stride,
                                       conv_dims,
                                       kernel_size=kernel_size,
                                       conv_norm=conv_norm,
                                       bias_fill=True,
                                       deformable_on=deformable_on,
                                       dilation=self.raf_dilation,
                                       bottleneck_channels=bottleneck_channels,
                                       activation=None
                                       )
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)

    def forward(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        losses = {}
        if self.training:
            assert targets
        # features should be tuple of 2 elements, first for center, second for raf
        cls = self.cls_head(features[0])
        wh = self.wh_head(features[0])
        reg = self.reg_head(features[0])
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.relation_on:
            rafs = self.raf_head(features[1])
            preds.update({'raf': rafs})

        if self.training:
            losses.update(self.loss(preds, targets))
            if self.relation_on:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})

        return losses, preds


@HEADS_REGISTRY.register()
class CenternetMultiStageHeads(GeneralHeads):
    """
    There will be several stages in the prediction head, such that centers and raf are predicted first,
    then the features are concat and feed to next stage for prediction refinement.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        assert self.relation_on, "to use this head, RELATION.RELATION_ON has to be set to True."
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.num_stages = cfg.MODEL.HEADS.NUM_STAGES
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dims = [conv_dim] * num_conv
        assert len(self.in_features) == 1 and len(self.in_channels) == 1
        self.output_stride = cfg.MODEL.HEADS.OUTPUT_STRIDES[0]
        self.stages = []
        in_channels = self.in_channels + [self.num_classes + self.num_predicates * 2] * (self.num_stages - 1)
        cls_out_channels = [self.num_classes] * self.num_stages
        raf_out_channels = [self.num_predicates * 2] * self.num_stages
        for i in range(self.num_stages):
            cls_head = SingleHead(
                            in_channels[i],
                            cls_out_channels[i],
                            self.output_stride,
                            conv_dims,
                            conv_norm=conv_norm,
                            bias_fill=True,
                            bias_value=self.cls_bias_value,
                            deformable_on=False,
                            bottleneck_channels=bottleneck_channels
            )
            raf_head = SingleHead(
                            in_channels[i],
                            raf_out_channels[i],
                            self.output_stride,
                            conv_dims,
                            kernel_size=7,
                            conv_norm=conv_norm,
                            bias_fill=not conv_norm,
                            deformable_on=deformable_on,
                            dilation=self.raf_dilation,
                            bottleneck_channels=bottleneck_channels
            )
            cls_head_name = "cls_head_" + str(i)
            raf_head_name = "raf_head_" + str(i)
            self.add_module(cls_head_name, cls_head)
            self.add_module(raf_head_name, raf_head)
            self.stages.append((cls_head, raf_head))
        # only last stage has wh and reg heads
        self.wh_head = SingleHead(in_channels[-1], 2, self.output_stride,
                                  conv_dims, conv_norm=conv_norm, activation=None)
        self.reg_head = SingleHead(in_channels[-1], 2, self.output_stride,
                                   conv_dims, conv_norm=conv_norm, activation=None)
        if self.training:
            self.raf_loss_evaluator = RAFLoss(cfg)

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:

        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]

        features = [features[f] for f in self.in_features][0]
        preds = defaultdict(list)
        for i, (cls_head, raf_head) in enumerate(self.stages):
            cls = cls_head(features)
            raf = raf_head(features)
            preds['cls'].append(cls)
            preds['raf'].append(raf)
            if i < len(self.stages) - 1:
                features = torch.cat((cls, raf), dim=1)
        preds['reg'].append(self.reg_head(features))
        preds['wh'].append(self.wh_head(features))

        if self.training:
            losses.update(self.loss(preds, targets))
        else:
            # only return the last prediction
            preds = {k: v[-1] for k, v in preds.items()}
            preds["cls"] = torch.sigmoid(preds["cls"])
            preds["raf"] = torch.tanh(preds["raf"])

        return losses, preds

    def loss(self,
             predictions: Dict[str, List[torch.Tensor]],
             targets: List[SceneGraph]) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-stage loss.
        """
        pred_scores = predictions['cls'] # list
        rafs = predictions['raf']

        num_instances = [len(x) for x in targets]

        # multi-use index, dim0 - image index in B, dim1 - class index in center maps, dim2 - spatial loc of center
        index = [torch.stack((torch.ones(num_instances[i], dtype=torch.long, device=x.gt_index.device) * i,
                              x.gt_classes,
                              x.gt_index))
                 for i, x in enumerate(targets)]

        index = cat(index, dim=1)
        loss = {}
        for i, (pred_score, raf) in enumerate(zip(pred_scores, rafs)):
            pred_score = torch.sigmoid(pred_score)
            raf = torch.tanh(raf)
            loss.update({"loss_cls_" + str(i):
                             self.hm_loss_func(pred_score, targets) * self.ct_loss_weight})
            loss.update({"loss_raf_" + str(i): self.raf_loss_evaluator(raf, targets) * self.raf_loss_weight})


        gt_wh = cat([x.gt_wh for x in targets], dim=0)
        gt_reg = cat([x.gt_reg for x in targets], dim=0)
        # width and height loss, better version
        loss_wh = centernet_utils.reg_l1_loss(predictions['wh'][0], index, gt_wh)

        # regression loss
        loss_reg = centernet_utils.reg_l1_loss(predictions['reg'][0], index, gt_reg)

        # loss_cls *= (self.ct_loss_weight / self.num_stages)
        # loss_raf *= (self.raf_loss_weight / self.num_stages)

        loss.update({
            "loss_box_wh": loss_wh * self.wh_loss_weight,
            "loss_center_reg": loss_reg * self.reg_loss_weight,
        })
        return loss

def build_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.HEADS.NAME
    return HEADS_REGISTRY.get(name)(cfg, input_shape).freeze()