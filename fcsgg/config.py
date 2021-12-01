"""
Extra configurations for FCSGG.

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

from detectron2.config import CfgNode as CN

def add_fcsgg_config(cfg: CN):
    """
    Add config for fcsgg.
    """
    _C = cfg

    _C.INPUT.LOAD_RELATIONS = False
    _C.INPUT.CENTER_GAUSSIAN_IOU_MIN_OVERLAP = 0.5
    _C.INPUT.CENTER_GAUSSIAN_IOU_MIN_OVERLAPS = [0.5]
    _C.INPUT.ADD_IGNORE = False
    _C.INPUT.GT_SCALE_AWARE = True
    _C.INPUT.GT_REGRESSION_SCALE_AWARE = False
    _C.INPUT.FLIP = True
    _C.INPUT.RAF_STRIDE_RATIO = 1
    _C.INPUT.RAF_TYPE = "vector" # or "point"
    _C.INPUT.USE_FIX_SIZE = False

    # relation
    _C.RELATION = CN()
    _C.RELATION.RELATION_ON = False
    _C.RELATION.RELATION_ONLY = False
    _C.RELATION.USE_FREQ_BIAS = False
    _C.RELATION.USE_GT_BOX = False # for evaluation only
    _C.RELATION.USE_GT_OBJECT_LABEL = False # for evaluation only
    _C.RELATION.MULTIPLE_PREDS = True
    _C.RELATION.IOU_THRESHOLD = 0.5


    # model

    _C.MODEL.META_ARCHITECTURE = "CenterNet"
    _C.MODEL.SIZE_DIVISIBILITY = 32
    _C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    _C.MODEL.BACKBONE.FREEZE_ALL = False # for support freezing fpn
    _C.MODEL.BACKBONE.WEIGHTS = "" # for support centernet weights
    _C.MODEL.RESNETS.DEPTH = 50
    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    _C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.RESNETS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1

    # biFPN
    _C.MODEL.FPN.REPEAT = 1
    _C.MODEL.FPN.NUM_BRANCHES = 1
    # DLA backbone
    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.FREEZE = False
    _C.MODEL.DLA.WEIGHTS = ""
    _C.MODEL.DLA.CONV_BODY = "DLA34"
    _C.MODEL.DLA.NORM = "BN"
    _C.MODEL.DLA.LEVELS = [1, 1, 1, 2, 2, 1]
    _C.MODEL.DLA.CHANNELS = [16, 32, 64, 128, 256, 512]
    # Hourglass backbone
    _C.MODEL.HOURGLASS = CN()
    _C.MODEL.HOURGLASS.OUT_FEATURES = ["hg_stack_1"]
    _C.MODEL.HOURGLASS.NORM = "SyncBN"
    _C.MODEL.HOURGLASS.NUM_STACKS = 8
    _C.MODEL.HOURGLASS.NUM_FEATURES = 256
    _C.MODEL.HOURGLASS.NUM_BLOCKS = [3]
    # Triad backbone
    _C.MODEL.TRIAD = CN()
    _C.MODEL.TRIAD.OUT_FEATURES = ["decoder_0", "decoder_1"]
    _C.MODEL.TRIAD.NORM = "SyncBN"
    _C.MODEL.TRIAD.NUM_BRANCHES = 2
    _C.MODEL.TRIAD.NUM_FEATURES = 256
    _C.MODEL.TRIAD.ENCODER_BLOCKS = 4
    _C.MODEL.TRIAD.DECODER_BLOCKS = [2, 2]
    # For HigherHRNet w32
    _C.MODEL.HRNET = CN()
    _C.MODEL.HRNET.UPSAMPLE_MODE = "nearest"
    _C.MODEL.HRNET.NORM = "SyncBN"
    _C.MODEL.HRNET.BN_MOMENTUM = 0.01
    _C.MODEL.HRNET.STEM_INPLANES = 64
    _C.MODEL.HRNET.STAGE2 = CN()
    _C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
    _C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
    _C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE3 = CN()
    _C.MODEL.HRNET.STAGE3.NUM_MODULES = 4
    _C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    _C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
    _C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE4 = CN()
    _C.MODEL.HRNET.STAGE4.NUM_MODULES = 3
    _C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    _C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    _C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.FINAL_STAGE_MULTISCALE = False
    _C.MODEL.HRNET.OUT_FEATURES = ["hr2", "hr3", "hr4", "hr5"]
    _C.MODEL.HRNET.WEIGHTS = "" # for load weights from other pretrained models

    _C.MODEL.NECKS = CN()
    _C.MODEL.NECKS.ENABLED = True
    _C.MODEL.NECKS.NAME = "Res5UpSampleNeck"
    _C.MODEL.NECKS.IN_FEATURES = ["res5"]
    _C.MODEL.NECKS.OUT_FEATURES = ["upsample_2"]
    _C.MODEL.NECKS.FREEZE = False # only support freeze all of a single neck
    _C.MODEL.NECKS.NORM = "BN"
    _C.MODEL.NECKS.MOMENTUM = 0.01
    _C.MODEL.NECKS.DEFORM_ON_PER_STAGE = [True, True, True]
    _C.MODEL.NECKS.WIDTH_PER_GROUP = 64
    _C.MODEL.NECKS.DEFORM_MODULATED = True
    _C.MODEL.NECKS.NUM_GROUPS = 1
    _C.MODEL.NECKS.DEFORM_NUM_GROUPS = 1
    _C.MODEL.NECKS.OUT_CHANNELS = [256, 128, 64] # for each upconv layer
    _C.MODEL.NECKS.DECONV_KERNEL_SIZES = [4, 4, 4]
    _C.MODEL.NECKS.UPSAMPLE_MODE = "nearest"
    _C.MODEL.NECKS.POOLING = "MAX"
    _C.MODEL.NECKS.NUM_OUTS = 4
    _C.MODEL.NECKS.CONV_STRIDE = 1
    _C.MODEL.NECKS.OUT_STRIDES = [4, 8, 16, 32]
    _C.MODEL.NECKS.TRIDENT = CN()
    _C.MODEL.NECKS.TRIDENT.NUM_BRANCH = 3
    _C.MODEL.NECKS.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
    _C.MODEL.NECKS.TRIDENT.BOTTLENECK_CHANNELS = 128
    _C.MODEL.NECKS.TRIDENT.BLOCK_OUT_CHANNELS = 256




    _C.MODEL.HEADS = CN()
    _C.MODEL.HEADS.FREEZE = [] # ["cls_head", "wh_head", "reg_head"]
    _C.MODEL.HEADS.NAME = "CenternetHeads"
    _C.MODEL.HEADS.IN_FEATURES = []
    _C.MODEL.HEADS.NUM_CLASSES = 150
    _C.MODEL.HEADS.NUM_PREDICATES = 50
    _C.MODEL.HEADS.BN_MOMENTUM = 0.01
    # hard-coded the dimension of the feature map size respect to input image
    _C.MODEL.HEADS.OUTPUT_STRIDE = 4
    _C.MODEL.HEADS.SHARED = False
    _C.MODEL.HEADS.OUTPUT_STRIDES = [4]
    _C.MODEL.HEADS.CLS_BIAS_VALUE = -2.19
    _C.MODEL.HEADS.CONV_DIM = 64
    _C.MODEL.HEADS.NUM_STAGES = 1
    _C.MODEL.HEADS.NUM_CONV = 2
    _C.MODEL.HEADS.LAST_DEFORM_ON = False
    _C.MODEL.HEADS.NUM_GROUPS = 1
    _C.MODEL.HEADS.WIDTH_PER_GROUP = 64
    _C.MODEL.HEADS.NORM = "GN"
    _C.MODEL.HEADS.LOSS = CN()
    _C.MODEL.HEADS.LOSS.HEATMAP_LOSS_TYPE = "efficient_focal_loss"
    _C.MODEL.HEADS.LOSS.CT_WEIGHT = 1.0
    _C.MODEL.HEADS.LOSS.WH_WEIGHT = 1.0
    _C.MODEL.HEADS.LOSS.REG_WEIGHT = 1.0
    _C.MODEL.HEADS.LOSS.RAF_WEIGHT = 1.0
    # Relation Affinity Field
    _C.MODEL.HEADS.RAF = CN()
    _C.MODEL.HEADS.RAF.LAST_DEFORM_ON = False
    _C.MODEL.HEADS.RAF.NUM_GROUPS = 1
    _C.MODEL.HEADS.RAF.WIDTH_PER_GROUP = 64
    _C.MODEL.HEADS.RAF.KERNEL_SIZE = 7
    _C.MODEL.HEADS.RAF.DILATION = 1
    _C.MODEL.HEADS.RAF.NON_LOCAL = False
    _C.MODEL.HEADS.RAF.CONV_DIM = 64
    _C.MODEL.HEADS.RAF.CONV_DIMS = []
    _C.MODEL.HEADS.RAF.NUM_CONV = 4
    _C.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO = 1
    _C.MODEL.HEADS.RAF.SPLIT = False
    _C.MODEL.HEADS.RAF.ADD_COORD = False
    _C.MODEL.HEADS.RAF.ADD_RELATIONESS = False

    _C.MODEL.HEADS.RAF.LOSS_TYPE = ("l1", "all", "normal", 1.0) # <l1, l2> <pos, all> <normal, cb> <int>


