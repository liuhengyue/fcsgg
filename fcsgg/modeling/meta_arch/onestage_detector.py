"""
One-stage detector and CenterNet implementations.

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Xingyi Zhou https://github.com/xingyizhou/CenterNet"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..heads import build_heads
from ..necks import build_necks

from fcsgg.structures import SceneGraph
from fcsgg.utils.centernet_utils import detector_postprocess, CenterNetDecoder
from fcsgg.data.detection_utils import GroundTruthGen
__all__ = ["CenterNet"]


@META_ARCH_REGISTRY.register()
class OneStageDetector(nn.Module):
    """
    Generalized one stage detector. Any models that contains the following 2 components:
    1. Per-image feature extraction (aka backbone)
    2 (optional). Neck modules
    3. Prediction head(s)
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: nn.Module,
        necks: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        # hack the size size_divisibility
        self.backbone._size_divisibility = 32
        self.necks = necks
        self.heads = heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "scene_graph" in batched_inputs[0]:
            gt_scene_graph = [x["scene_graph"].to(self.device) for x in batched_inputs]
        else:
            gt_scene_graph = None

        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        detector_losses = self.heads(features, gt_scene_graph)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        if detected_instances is None:

            _, results = self.heads(features)
        else:
            results = None
            # detected_instances = [x.to(self.device) for x in detected_instances]
            # results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # the default results should be List[Instances], each Instances
        # should at least have fields of pred_boxes, scores and pred_classes.
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        raise NotImplementedError()


@META_ARCH_REGISTRY.register()
class CenterNet(OneStageDetector):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            heads: nn.Module,
            necks: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            output_stride: int = 4,
            size_divisibility: int = 32,
            pretrained_weights: str = None,
            relation_on: bool = False,
            use_gt_box: bool = False,
            use_gt_object_label: bool = False,
            add_ignore: bool = False,
            output_strides: [4],
            raf_stride_ratio: int = 1,
            gt_gen: GroundTruthGen = None,
            raf_type: str = "vector",
            use_fix_size: bool = False,
            max_size_test: int = 1024,
            relation_only: bool = False,
            use_freq_bias: bool = False,

    ):
        super().__init__(backbone=backbone, heads=heads, necks=necks, pixel_mean=pixel_mean,
                         pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)

        self.output_stride = output_stride
        self.size_divisibility = size_divisibility
        self.relation_on = relation_on
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.add_ignore = add_ignore
        self.output_strides = output_strides
        self.raf_type = raf_type
        self.postprocessor = CenterNetDecoder(self.relation_on,
                                              self.use_gt_box,
                                              self.use_gt_object_label,
                                              output_strides=output_strides,
                                              raf_stride_ratio=raf_stride_ratio,
                                              raf_type=self.raf_type,
                                              use_fix_size=use_fix_size,
                                              max_size_test=max_size_test,
                                              relation_only=relation_only,
                                              freq_bias=use_freq_bias)
        self.gt_gen = gt_gen
        if pretrained_weights:
            self.pretrained_weights = pretrained_weights
            self.load_pretrained_model()

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # override size_divisibility since we are encoder-decoder type of archs
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images




    def preprocess_gt(self, gt_scene_graphs, image_size, image_ids):
        """
        Normalize, pad and batch the input images (if necessary)
        """

        # due to detectron2 resize image tensors using imagelist, follow the same way by resizing the
        # heatmap, and put back to the corresponding image (not too necessary)
        for i, (x, image_id) in enumerate(zip(gt_scene_graphs, image_ids)):
            gt_scene_graphs[i] = self.gt_gen(x, image_size, image_id, training=self.training)




    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * scene_graph (optional): groundtruth :class:`SceneGraph`

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "scene_graph" in batched_inputs[0]:
            gt_scene_graphs = [x["scene_graph"].to(self.device) for x in batched_inputs]
            image_ids = [x["image_id"] for x in batched_inputs]
            self.preprocess_gt(gt_scene_graphs, images.tensor.shape[-2:], image_ids)

        else:
            gt_scene_graphs = None

        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        detector_losses, _ = self.heads(features, gt_scene_graphs)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        if "scene_graph" in batched_inputs[0]:
            gt_scene_graphs = [x["scene_graph"].to(self.device) for x in batched_inputs]
            image_ids = [x["image_id"] for x in batched_inputs]
            self.preprocess_gt(gt_scene_graphs, images.tensor.shape[-2:], image_ids)
            # for `decode` purpose
            for inputs, gt_scene_graph in zip(batched_inputs, gt_scene_graphs):
                inputs["scene_graph"] = gt_scene_graph
        else:
            gt_scene_graphs = None

        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        if detected_instances is None:

            _, results = self.heads(features, gt_scene_graphs)
        else:
            results = None

        # the default results should be List[Instances], each Instances
        # should at least have fields of pred_boxes, scores and pred_classes.
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        detections = self.postprocessor.decode(instances, batched_inputs, image_sizes, self.device)
        for detection, input_per_image, image_size in zip(
                detections, batched_inputs, image_sizes
        ):
            # augmented image size without padding (stored in `images.image_sizes`)
            # to the original image size
            detection["input_height"] = image_size[0]
            detection["input_width"] = image_size[1]
            detection["output_height"] = input_per_image.get("height", image_size[0])
            detection["output_width"] = input_per_image.get("width", image_size[1])
            sg_r, r = detector_postprocess(detection, relation_on=self.relation_on)
            processed_results.append({"scene_graph": sg_r, "instances": r})
        return processed_results

    def load_pretrained_model(self):
        # for loading the DLA models
        import torch.utils.model_zoo as model_zoo
        from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
        model_weights = model_zoo.load_url(self.pretrained_weights)
        model_weights = model_weights["state_dict"]
        # convert names
        processed_dict = {}
        for k, v in model_weights.items():
            k = k[7:] # remove 'module.'
            if "base." in k:
                k = k.replace("base.", "backbone.")
                processed_dict[k] = v
            if "ida_up" in k or "dla_up" in k:
                k = "necks." + k
                processed_dict[k] = v

        own_state = self.state_dict()
        align_and_update_state_dicts(own_state, processed_dict, c2_conversion=False)

        self.load_state_dict(own_state)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # detectron2 fpn does not have fpn freeze, so do it here
        # IMPORTANT: For HRNet, it does not work! Unless, we make the HRNet convs replaced with Detectron2 conv.
        if cfg.MODEL.BACKBONE.FREEZE_ALL:
            from detectron2.layers import FrozenBatchNorm2d
            for p in backbone.parameters():
                p.requires_grad = False
        necks = build_necks(cfg, backbone.output_shape())
        if necks:
            heads = build_heads(cfg, necks.output_shape())
        else:
            heads = build_heads(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "output_stride": cfg.MODEL.HEADS.OUTPUT_STRIDE,
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "pretrained_weights": cfg.MODEL.BACKBONE.WEIGHTS, # different from MODEL.WEIGHTS
            "relation_on": cfg.RELATION.RELATION_ON,
            "use_gt_box": cfg.RELATION.USE_GT_BOX,
            "use_gt_object_label": cfg.RELATION.USE_GT_OBJECT_LABEL,
            "add_ignore": cfg.INPUT.ADD_IGNORE,
            "output_strides": cfg.MODEL.HEADS.OUTPUT_STRIDES,
            "raf_stride_ratio": cfg.INPUT.RAF_STRIDE_RATIO,
            "gt_gen": GroundTruthGen(cfg),
            "raf_type": cfg.INPUT.RAF_TYPE,
            "use_fix_size": cfg.INPUT.USE_FIX_SIZE,
            "max_size_test": cfg.INPUT.MAX_SIZE_TEST,
            "relation_only": cfg.RELATION.RELATION_ONLY,
            "use_freq_bias": cfg.RELATION.USE_FREQ_BIAS,
        }

