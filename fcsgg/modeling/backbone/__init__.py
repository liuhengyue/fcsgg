from .resnet import build_resnet_fpn_p2_backbone
from .dla import DLA, build_dla_backbone, build_dla_fpn_backbone, build_fcos_dla_fpn_backbone
from .hourglass import build_hourglass_net, build_dual_decoder_hourglass_net
from .hrnet import build_hrnet_backbone
from .bifpn import build_resnet_bifpn_backbone
from .dual_fpn import build_resnet_dual_bifpn_backbone
from .triad import build_triad_net