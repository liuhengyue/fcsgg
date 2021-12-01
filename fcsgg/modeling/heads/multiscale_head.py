"""
Core implementations of detection heads in multi-scale.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = [""]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

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
from detectron2.layers import cat, Conv2d, get_norm, CNNBlockBase
import fvcore.nn.weight_init as weight_init
from fcsgg.structures import SceneGraph

import fcsgg.utils.centernet_utils as centernet_utils
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from .heads import *
from fcsgg.layers import MultiscaleFusionLayer, NONLocalBlock2D, NONLocalBlock3D, MultiscaleSum
from fcsgg.modeling.losses import RAFLoss
HEADS_REGISTRY.__doc__ = """
Registry for a multiscale-head in a single-stage model.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Head`.
"""

logger = logging.getLogger(__name__)


def interpolate_sum(features: List[torch.Tensor]):
    out_features = features[0]
    for i in range(1, len(features)):
        out_features += F.interpolate(features[i], scale_factor=2 ** i, mode='bilinear', align_corners=True)
    return out_features



@HEADS_REGISTRY.register()
class MultiScaleHeads(CenternetRelationHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # self.head = CenternetRelationHeads(cfg, input_shape)
        # the strides to downsample the gt, [1, 2, 4, 8]
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) == len(self.in_features), "number of features and strides mismatch."
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES


    def _forward(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(features):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_stride = "_" + str(self.in_features[i])
            loss, pred = self._forward(features_per_scale, targets_per_scale)
            loss = {k + cur_stride: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds

@HEADS_REGISTRY.register()
class MultiScaleSwitchNormHeads(SwitchNormHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) == len(self.in_features), "number of features and strides mismatch."
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES


    def _forward(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets, stride)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(features):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward(features_per_scale, targets_per_scale, self.strides[i])
            loss = {k + cur_stride: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds


@HEADS_REGISTRY.register()
class MultiScaleSwitchNormDAHeads(GeneralHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert not self.single_scale
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.relation_on = cfg.RELATION.RELATION_ON
        self.use_gt_label = cfg.RELATION.USE_GT_OBJECT_LABEL
        self.use_non_local = cfg.MODEL.HEADS.RAF.NON_LOCAL
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        self.fpn_num_branches = cfg.MODEL.FPN.NUM_BRANCHES
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        self.num_levels = len(self.output_strides)
        bn_momentum = cfg.MODEL.HEADS.BN_MOMENTUM
        # the strides to downsample the gt, [1, 2, 4, 8]
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) * self.fpn_num_branches == len(self.in_features), \
            "number of features and strides mismatch."
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        raf_conv_dims = [cfg.MODEL.HEADS.RAF.CONV_DIM] * cfg.MODEL.HEADS.RAF.NUM_CONV
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE

        in_channels = self.in_channels[0]
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
                                              bn_momentum=bn_momentum
                                              )

            self.attn_layer = DA_Module(self.num_classes, self.num_predicates, down_ratio=self.down_ratio)
            self.aggr_layer = MultiscaleSum(len(self.output_strides))

            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)


    def _forward_raf_branch(
            self,
            features: torch.Tensor,
            obj_features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: should be tuple of 2 elements, first for center, second for raf
        """
        losses = {}
        if self.training:
            assert targets
        rafs = self.raf_head(features, stride)
        rafs = self.attn_layer(obj_features, rafs)
        preds = {'raf': rafs}
        if self.training:
            loss_raf = self.raf_loss_evaluator(rafs, targets)
            loss_raf *= self.raf_loss_weight
            losses = {'loss_raf': loss_raf}

        return losses, preds

    def _forward_object_branch(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: center branch features from backbone (w/fpn)
        """
        losses = {}
        if self.training:
            assert targets
        cls = self.cls_head(features, stride)
        wh = self.wh_head(features, stride)
        reg = self.reg_head(features, stride)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.training:
            losses.update(self.loss(preds, targets, after_activation=True))
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        if self.fpn_num_branches == 2:
            # right now only support 2 branches
            object_branch_features = features[:self.num_levels]
            relation_branch_features = features[self.num_levels:]
        else:
            object_branch_features = features
            relation_branch_features = features
        center_preds = []
        targets_all_scales = [[target[i] for target in targets] for i in range(len(self.output_strides))] \
            if self.training else [None for _ in range(len(self.output_strides))]
        # forward object branch first
        for i, features_per_scale in enumerate(object_branch_features):
            # form a SceneGraph
            targets_per_scale = targets_all_scales[i]
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward_object_branch(features_per_scale, targets_per_scale, self.strides[i])
            loss = {k + cur_stride: v for k, v in loss.items()}
            # if self.training or self.use_gt_label:
            #     ct_maps_per_scale = torch.stack([target.gt_ct_maps[i] for target in targets], dim=0)
            # else:
            # # pred['cls'] = torch.sigmoid(pred['cls'])
            #     ct_maps_per_scale = pred['cls'].detach()
            ct_maps_per_scale = pred['cls'].detach()
            losses.update(loss)
            center_preds.append(ct_maps_per_scale)
            preds.append(pred)
        if self.relation_on:
            center_features = self.aggr_layer(center_preds)
            # then forward raf branch
            for i, features_per_scale in enumerate(relation_branch_features):
                # add non local block
                # nonlocal_feat = self.non_local_attn(fused_center_feats[i])
                # cat_feat = torch.cat((features_per_scale, nonlocal_feat), dim=1)
                # if self.use_non_local:
                #     features_per_scale = self.non_local_attn(fused_center_feats[i], features_per_scale)
                # else:
                # simple concat
                # features_per_scale = torch.cat((features_per_scale, center_preds[i]), dim=1)
                targets_per_scale = targets_all_scales[i]
                cur_stride = "_s" + str(self.strides[i])
                loss, pred = self._forward_raf_branch(features_per_scale,
                                                      center_features[i],
                                                      targets_per_scale,
                                                      self.strides[i])
                loss = {k + cur_stride: v for k, v in loss.items()}
                losses.update(loss)
                preds[i].update(pred)
        return losses, preds


@HEADS_REGISTRY.register()
class MultiScaleSwitchNormConcatClassHeads(GeneralHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert not self.single_scale
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.relation_on = cfg.RELATION.RELATION_ON
        self.use_gt_label = cfg.RELATION.USE_GT_OBJECT_LABEL
        self.use_non_local = cfg.MODEL.HEADS.RAF.NON_LOCAL
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        self.fpn_num_branches = cfg.MODEL.FPN.NUM_BRANCHES
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        split_pred = cfg.MODEL.HEADS.RAF.SPLIT
        bn_momentum = cfg.MODEL.HEADS.BN_MOMENTUM
        # the strides to downsample the gt, [1, 2, 4, 8]
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) * self.fpn_num_branches == len(self.in_features), \
            "number of features and strides mismatch."
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        raf_conv_dims = [cfg.MODEL.HEADS.RAF.CONV_DIM] * cfg.MODEL.HEADS.RAF.NUM_CONV
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE

        in_channels = self.in_channels[0]
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
            bn_momentum=bn_momentum,
            output_feat=True
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
            self.raf_head = MultiBNSingleHead(in_channels + conv_dims[-1],
                                              2 * self.num_predicates,
                                              self.output_stride,
                                              raf_conv_dims,
                                              kernel_size=kernel_size,
                                              conv_norm=conv_norm,
                                              bias_fill=True,
                                              deformable_on=deformable_on,
                                              dilation=self.raf_dilation,
                                              bottleneck_channels=bottleneck_channels,
                                              activation=torch.tanh_,
                                              output_strides=self.output_strides,
                                              down_ratio=self.down_ratio,
                                              up_ratio=1,
                                              split_pred=split_pred
                                              )
            self.aggr_layer = MultiscaleSum(len(self.output_strides), aggr_method="sum")

            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)


    def _forward_raf_branch(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: should be tuple of 2 elements, first for center, second for raf
        """
        losses = {}
        if self.training:
            assert targets
        rafs = self.raf_head(features, stride)
        preds = {'raf': rafs}
        if self.training:
            loss_raf = self.raf_loss_evaluator(rafs, targets)
            loss_raf *= self.raf_loss_weight
            losses = {'loss_raf': loss_raf}

        return losses, preds

    def _forward_object_branch(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: center branch features from backbone (w/fpn)
        """
        losses = {}
        if self.training:
            assert targets
        cls, obj_feat = self.cls_head(features, stride)
        wh = self.wh_head(features, stride)
        reg = self.reg_head(features, stride)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg,
            'obj_feat': obj_feat
        }
        if self.training:
            losses.update(self.loss(preds, targets, after_activation=True))
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        if self.fpn_num_branches == 2:
            # right now only support 2 branches
            object_branch_features = features[:len(self.output_strides)]
            relation_branch_features = features[len(self.output_strides):]
        else:
            object_branch_features = features
            relation_branch_features = features
        center_preds = []
        targets_all_scales = [[target[i] for target in targets] for i in range(len(self.output_strides))] \
            if self.training else [None for _ in range(len(self.output_strides))]
        # forward object branch first
        for i, features_per_scale in enumerate(object_branch_features):
            # form a SceneGraph
            targets_per_scale = targets_all_scales[i]
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward_object_branch(features_per_scale, targets_per_scale, self.strides[i])
            loss = {k + cur_stride: v for k, v in loss.items()}
            # if self.training or self.use_gt_label:
            #     ct_maps_per_scale = torch.stack([target.gt_ct_maps[i] for target in targets], dim=0)
            # else:
            ct_maps_per_scale = pred['obj_feat'].clone().detach()
            losses.update(loss)
            center_preds.append(ct_maps_per_scale)
            preds.append(pred)
        if self.relation_on:
            center_features = self.aggr_layer(center_preds)
            # then forward raf branch
            for i, features_per_scale in enumerate(relation_branch_features):
                features_per_scale = torch.cat((features_per_scale, center_features[i]), dim=1)
                targets_per_scale = targets_all_scales[i]
                cur_stride = "_s" + str(self.strides[i])
                loss, pred = self._forward_raf_branch(features_per_scale, targets_per_scale, self.strides[i])
                loss = {k + cur_stride: v for k, v in loss.items()}
                losses.update(loss)
                preds[i].update(pred)
        return losses, preds


@HEADS_REGISTRY.register()
class MultiScaleSwitchNormDualHeads(DualBranchSwitchNormHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.num_features_per_branch = len(self.strides)
        # the strides to downsample the gt, [4, 8, ...]
        assert self.num_features_per_branch * 2 == len(self.in_features), "number of features and strides mismatch."




    def _forward(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets, stride)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(zip(features[:self.num_features_per_branch],
                                                   features[self.num_features_per_branch:])):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward(features_per_scale, targets_per_scale, self.strides[i])
            loss = {k + cur_stride: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds


@HEADS_REGISTRY.register()
class MultiScaleSwitchNormAttnHeads(GeneralHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert not self.single_scale
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.relation_on = cfg.RELATION.RELATION_ON
        self.use_gt_label = cfg.RELATION.USE_GT_OBJECT_LABEL
        self.use_non_local = cfg.MODEL.HEADS.RAF.NON_LOCAL
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.output_stride = self.output_strides[0]
        self.fpn_num_branches = cfg.MODEL.FPN.NUM_BRANCHES
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        bn_momentum = cfg.MODEL.HEADS.BN_MOMENTUM
        # the strides to downsample the gt, [1, 2, 4, 8]
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) * self.fpn_num_branches == len( self.in_features), \
            "number of features and strides mismatch."
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        raf_conv_dims = [cfg.MODEL.HEADS.RAF.CONV_DIM] * cfg.MODEL.HEADS.RAF.NUM_CONV
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE


        in_channels = self.in_channels[0]
        self.cls_head = MultiBNSingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=conv_norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=None,
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
            self.raf_head = MultiBNSingleHead(in_channels + self.num_classes,
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
                                              down_ratio=self.down_ratio
                                              )

            self.fuse_layer = MultiscaleFusionLayer(len(self.output_strides),
                                                    [self.num_classes] * len(self.output_strides),
                                                    multi_scale_output=True,
                                                    norm="SyncBN")
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)

            # if self.use_non_local:
            #     self.non_local_attn = NONLocalBlock2D(self.num_classes,
            #                                           out_channels=in_channels,
            #                                           inter_channels=in_channels,
            #                                           norm=conv_norm,
            #                                           sub_sample=False,
            #                                           g_given=True)



    def _forward_raf_branch(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: should be tuple of 2 elements, first for center, second for raf
        """
        losses = {}
        if self.training:
            assert targets
        rafs = self.raf_head(features, stride)
        preds = {'raf': rafs}
        if self.training:
            loss_raf = self.raf_loss_evaluator(rafs, targets)
            loss_raf *= self.raf_loss_weight
            losses = {'loss_raf': loss_raf}

        return losses, preds

    def _forward_object_branch(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None,
            stride: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features: center branch features from backbone (w/fpn)
        """
        losses = {}
        if self.training:
            assert targets
        cls = self.cls_head(features, stride)
        wh = self.wh_head(features, stride)
        reg = self.reg_head(features, stride)
        preds = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.training:
            losses.update(self.loss(preds, targets, after_activation=False))
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        if self.fpn_num_branches == 2:
            # right now only support 2 branches
            object_branch_features = features[:len(self.output_strides)]
            relation_branch_features = features[len(self.output_strides):]
        else:
            object_branch_features = features
            relation_branch_features = features
        center_preds = []
        targets_all_scales = [[target[i] for target in targets] for i in range(len(self.output_strides))]  \
            if self.training else [None for _ in range(len(self.output_strides))]
        # forward object branch first
        for i, features_per_scale in enumerate(object_branch_features):
            # form a SceneGraph
            targets_per_scale = targets_all_scales[i]
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward_object_branch(features_per_scale, targets_per_scale, self.strides[i])
            loss = {k + cur_stride: v for k, v in loss.items()}
            # if self.training or self.use_gt_label:
            #     ct_maps_per_scale = torch.stack([target.gt_ct_maps[i] for target in targets], dim=0)
            # else:
            ct_maps_per_scale = pred['cls'].detach()
            center_preds.append(ct_maps_per_scale)
            losses.update(loss)
            pred['cls'] = torch.sigmoid(pred['cls'])
            preds.append(pred)
        if self.relation_on:
            # multi-scale center feature maps fusion
            fused_center_feats = self.fuse_layer(center_preds)
            # then forward raf branch
            for i, features_per_scale in enumerate(relation_branch_features):
                # add non local block
                # nonlocal_feat = self.non_local_attn(fused_center_feats[i])
                # cat_feat = torch.cat((features_per_scale, nonlocal_feat), dim=1)
                # if self.use_non_local:
                #     features_per_scale = self.non_local_attn(fused_center_feats[i], features_per_scale)
                # else:
                # simple concat
                features_per_scale = torch.cat((features_per_scale, fused_center_feats[i]), dim=1)
                targets_per_scale = targets_all_scales[i]
                cur_stride = "_s" + str(self.strides[i])
                loss, pred = self._forward_raf_branch(features_per_scale, targets_per_scale, self.strides[i])
                loss = {k + cur_stride: v for k, v in loss.items()}
                losses.update(loss)
                preds[i].update(pred)
        return losses, preds

@HEADS_REGISTRY.register()
class MultiScaleDualHeads(DualBranchHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) * 2 == len(self.in_features), "number of features and strides mismatch."
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES


    def _forward(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(zip(features[:4], features[4:])):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward(features_per_scale, targets_per_scale)
            loss = {k + cur_stride: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds

@HEADS_REGISTRY.register()
class MultiStageHeads(CenternetRelationHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) == len(self.in_features), "number of features and strides mismatch."
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.out_names = ["_s" + str(stride) + "_{}".format(i) for i, stride in enumerate(self.strides)]


    def _forward(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(features):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_name = self.out_names[i]
            loss, pred = self._forward(features_per_scale, targets_per_scale)
            loss = {k + cur_name: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds[-1]

@HEADS_REGISTRY.register()
class MultiScaleCascadeHeads(CenternetCascadeHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # self.head = CenternetRelationHeads(cfg, input_shape)
        # the strides to downsample the gt, [1, 2, 4, 8]
        assert len(cfg.MODEL.HEADS.OUTPUT_STRIDES) == len(self.in_features), "number of features and strides mismatch."
        self.strides = cfg.MODEL.HEADS.OUTPUT_STRIDES


    def _forward(
            self,
            features: torch.Tensor,
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        features:
        """
        losses, preds = super().forward(features, targets)
        return losses, preds

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            targets: Optional[List[SceneGraph]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = [features[f] for f in self.in_features]
        preds, losses = [], {}
        for i, features_per_scale in enumerate(features):
            # form a SceneGraph
            targets_per_scale = [target[i] for target in targets] if self.training else None
            cur_stride = "_s" + str(self.strides[i])
            loss, pred = self._forward(features_per_scale, targets_per_scale)
            loss = {k + cur_stride: v for k, v in loss.items()}
            losses.update(loss)
            preds.append(pred)
        return losses, preds