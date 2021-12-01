"""
Scene graph evaluator wrapper for Detectron2.

Modified to be compatible with Detectron2 based on
https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.evaluator import DatasetEvaluator
from fcsgg.data.datasets.visual_genome import convert_to_vg_json
from .sgg_evaluation import SceneGraphAPI

class VGEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        # looger parent is fcsgg, which is not set therefore get it from detectron2
        self._logger = logging.getLogger("detectron2.evaluation")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "sg_json_file"):
            self._logger.info(
                f"'{dataset_name}' has no json_file for scene graph."
                " Therefore trying to convert it to VG format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_sg_format.json")
            self._metadata.sg_json_file = cache_path
            convert_to_vg_json(dataset_name, cache_path, allow_cached=True)

        sg_json_file = PathManager.get_local_path(self._metadata.sg_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._sg_api = SceneGraphAPI(cfg, sg_json_file, logger=self._logger)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._sg_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ()
        if cfg.RELATION.RELATION_ON:
            if cfg.RELATION.USE_GT_BOX:
                tasks = tasks + ("sgcls",)
            elif cfg.RELATION.USE_GT_OBJECT_LABEL:
                tasks = tasks + ("predcls",)
            else:
                tasks = tasks + ("sgdet",)
        assert len(tasks) < 2, "For SGG evaluation, each task needs separate evaluation."
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            assert "scene_graph" in output, "No scene_graph is found in outputs."
            scene_graph = output["scene_graph"].to(self._cpu_device)
            prediction.update(scenegraph_to_json(scene_graph))

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[VGEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sg_predictions.pth")
            self._logger.info("Saving SGG predictions to {}".format(file_path))
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for VG format ...")

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "vg_sg_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            res = self._sg_api.evaluate(predictions, task)
            self._results[task] = res



def scenegraph_to_json(sg):
    """
    Dump an "SceneGraph" object to a dict that's used for evaluation.
    The output will be saved as json.

    Args:
        sg (SceneGraph):

    Returns:
        dict: contains predictions for one image.
    """
    boxes = sg.pred_boxes.tensor.numpy().tolist()
    # for vg evaluation, all boxes should be in XYXY_ABS
    scores = sg.scores.numpy().tolist()
    classes = sg.pred_classes.numpy().tolist()
    rel_scores = sg.rel_scores.numpy().tolist()
    rel_inds = sg.rel_inds.numpy().tolist()


    result = {
        "category_ids": classes,
        "bboxes": boxes,
        "scores": scores,
        "rel_scores": rel_scores,
        "rel_inds": rel_inds
    }
    return result



