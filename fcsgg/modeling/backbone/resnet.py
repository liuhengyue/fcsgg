"""
Simple ResNet FPN that only outputs p2.

Modified from https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation/blob/master/lib/models/pose_higher_hrnet.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = []
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY, build_resnet_backbone, FPN

class SingleFPN(FPN):
    """
    Basically only output one single scale feature. No top block
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", level="p2"
    ):
        super(SingleFPN, self).__init__(bottom_up, in_features, out_channels,
                                        norm=norm, top_block=top_block, fuse_type=fuse_type)
        assert level in self._out_features, "{} is not in the out_features list of FPN.".format(level)
        self.level = level
        self._out_feature_strides = {self.level: self._out_feature_strides[self.level]}
        self._out_features = [self.level]
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        del self.output_convs
        del self.fpn_output2, self.fpn_output3, self.fpn_output4

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
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        prev_features = self.lateral_convs[0](x[0])
        for features, lateral_conv in zip(
                x[1:], self.lateral_convs[1:]):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
        prev_features = self.fpn_output5(prev_features)
        return {self._out_features[0]: prev_features}



@BACKBONE_REGISTRY.register()
def build_resnet_fpn_p2_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = SingleFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone