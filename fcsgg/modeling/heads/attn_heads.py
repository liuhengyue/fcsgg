"""
Implementations of RAF head using convolutional attention CBAM.

Add attention modules seem not have significant impact on the performance. And yeah, it did not work out.

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

from .heads import *
from fcsgg.layers import MultiscaleFusionLayer, CBAM

class CBAMHead(CNNBlockBase):

    def __init__(self, in_channel, out_channel, stride,
                 conv_dims, dilation=1, bias_fill=False, bias_value=0,
                 conv_norm="", kernel_size=3,
                 deformable_on=False,
                 activation=None,
                 down_ratio=1,
                 up_sample=True):
        super(CBAMHead, self).__init__(in_channel, out_channel, stride)
        self.activation = activation
        self.conv_norm_relus = []
        self.attns = []
        self.down_ratio = down_ratio
        self.up_sample = up_sample
        cur_channels = in_channel

        for k, conv_dim in enumerate(conv_dims):
            stride = 2 if self.down_ratio > 1 and k < math.log2(self.down_ratio) else 1
            if deformable_on:
                conv = DeformBottleneckBlock(cur_channels, conv_dim,
                                             bottleneck_channels=conv_dim//2,
                                             stride=stride,
                                             norm=conv_norm,
                                             deform_modulated=True,
                                             deform_num_groups=1,
                                             dilation=1
            )
            else:
                conv = Conv2d(cur_channels, conv_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size * dilation - 1) // 2,
                              dilation=dilation,
                              bias=not conv_norm,
                              activation=F.relu,
                              norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM))
                attn = CBAM(conv_dim)

            self.add_module("head_fcn{}".format(k), conv)
            self.conv_norm_relus.append(conv)
            self.add_module("head_attn{}".format(k), attn)
            self.attns.append(attn)
            cur_channels = conv_dim

        self.out_conv = nn.Conv2d(cur_channels, out_channel, kernel_size=1)
        if not deformable_on:
            for layer in self.conv_norm_relus:
                weight_init.c2_msra_fill(layer)
        nn.init.xavier_normal_(self.out_conv.weight)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        for layer, attn in zip(self.conv_norm_relus, self.attns):
            x = layer(x)
            x = attn(x)
        if self.down_ratio > 1 and self.up_sample:
            x = F.interpolate(x,
                             scale_factor=self.down_ratio,
                             mode="nearest",
                             align_corners=None)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

@HEADS_REGISTRY.register()
class CenternetCBAMHead(CenternetHeads):
    """
    Extended heads with relation affinity field prediction head using CBAM.
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
        raf_conv_dims = cfg.MODEL.HEADS.RAF.CONV_DIMS
        if self.relation_on:
            self.raf_head = CBAMHead(
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
                                activation=torch.tanh_,
                                down_ratio=self.down_ratio
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
            rafs = self.raf_head(features)
            preds.update({'raf': rafs})
            if self.training:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({'loss_raf': loss_raf})
        return losses, preds

