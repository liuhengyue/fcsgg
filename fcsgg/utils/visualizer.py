"""
The customized scene graph visualizer to visualize RAFs using optical flow or arrows.

Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import colorsys
import logging
import math
from collections import Set
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import random_color, ColorMode, Visualizer, VisImage

logger = logging.getLogger(__name__)




def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class SceneGraphVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale=scale, instance_mode=instance_mode)
        self.scale = scale

        # we have mutliple outputs, first is the object detection
        # followed by P number of rafs with predicted relationship
        self.predicate_class_names = self.metadata.get("predicate_classes", None)
        self.object_class_names = self.metadata.get("thing_classes", None)
        assert self.predicate_class_names is not None
        assert self.object_class_names is not None
        self.object_colormap = mpl.pyplot.cm.get_cmap('PiYG', len(self.object_class_names))
        self.colormap = colormap(maximum=1.0)
        # add original image
        ori = VisImage(self.img, scale=self.scale)
        self.outputs = [ori, self.output]

    def _draw_single_prediction(self):
        pass

    def _draw_rafs(self, pred_graph, object_class_names, predicate_names):
        """
        Args:
            pred_graph

        Returns:
            list[str] or None
        """
        subj_labels, obj_labels, predicate_labels = None, None, None
        if object_class_names is not None and pred_graph is not None and pred_graph.shape[0] > 0:
            subj_classes, obj_classes, predicate_classes, \
            subj_scores, obj_scores, predicate_scores, subj_boxes, obj_boxes = \
                np.hsplit(pred_graph, [1, 2, 3, 4, 5, 6, 10])
            predicate_scores = predicate_scores[:, 0]
            subj_scores = subj_scores[:, 0]
            obj_scores = obj_scores[:, 0]
            subj_labels = [object_class_names[int(i)] for i in subj_classes]
            obj_labels = [object_class_names[int(i)] for i in obj_classes]
            predicate_labels = [predicate_names[int(i)] for i in predicate_classes]
        if predicate_scores is not None:
            if subj_labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in predicate_scores]
            else:
                labels = ["{} {:.0f}%     {} {:.0f}%     {} {:.0f}%".format(s_l, s_s * 100,
                                                                            p_l, p_s * 100,
                                                                            o_l, o_s * 100)
                          for s_l, p_l, o_l, s_s, o_s, p_s
                          in zip(subj_labels, predicate_labels, obj_labels,
                                 subj_scores, obj_scores, predicate_scores)]
                subj_centers = np.stack(((subj_boxes[:, 0] + subj_boxes[:, 2]) / 2,
                                         (subj_boxes[:, 1] + subj_boxes[:, 3]) / 2),
                                        axis=1)
                obj_centers = np.stack(((obj_boxes[:, 0] + obj_boxes[:, 2]) / 2,
                                        (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2),
                                       axis=1)
                locations = (subj_centers + obj_centers) / 2
                cmap = colormap(maximum=1.0)
                colors = [cmap[int(i)] for i in predicate_classes]

        return labels, locations, colors

    def draw_sg_predictions(self, predictions, T=0.2):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        keep = scores > T
        scores = scores[keep]
        boxes = boxes[keep]
        classes = classes[keep]
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        # sum of rafs from all scales
        rafs = predictions.pred_rafs if predictions.has("pred_rafs") else None
        pred_graph = predictions.pred_graph.numpy() if predictions.has("pred_graph") else None
        # text_labels, text_positions, text_colors = _create_triplet_labels(pred_graph,
        #                                 self.metadata.get("thing_classes", None),
        #                                 self.metadata.get("predicate_classes", None))

        # we do not have mask or segmentation
        masks = None
        colors = None
        alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.5

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        _, num_predicates = self.draw_rafs(rafs, pred_graph)

        return self.outputs, num_predicates

    def draw_rafs(self,
                  rafs,
                  pred_graph,
                  threshold=0.1):
        # record if a object has been label for each output
        objects_sets = [set() for _ in range(len(self.predicate_class_names))]
        # output_inds = set([0])
        # process pred_graph
        # compute existing predicates
        # existing_predicates = np.unique(pred_graph[:, 2]).astype(np.int32)
        # change it to all predicates
        existing_predicates = np.array([i for i in range(50)], dtype=np.int32)
        # init visimage
        outputs = {}
        rafs = rafs.permute(0, 2, 3, 1).numpy()
        for predicate_cls in existing_predicates:
            rafs_vis = flow_to_color(rafs[predicate_cls])
            rafs_vis = cv2.addWeighted(rafs_vis, 0.6, self.img, 0.4, 0)
            vis_fig = VisImage(rafs_vis, scale=self.scale)
            self.draw_text_on_target(
                vis_fig,
                self.predicate_class_names[predicate_cls],
                (vis_fig.width//2, 10),
                color='white',
                horizontal_alignment="center",
                font_size=self._default_font_size * 2,
                alpha=1.0
            )
            # vis_fig.ax.set_title(self.predicate_class_names[predicate_cls])
            outputs[predicate_cls] = vis_fig
        for i, pred in enumerate(pred_graph):
            # split into fields
            subj_cls, obj_cls, predicate_cls, \
            subj_score, obj_score, predicate_score, subj_box, obj_box,\
            subj_id, obj_id = \
                np.hsplit(pred, [1, 2, 3, 4, 5, 6, 10, 14, 15])
            if predicate_score < threshold:
                continue
            subj_cls = int(subj_cls)
            obj_cls = int(obj_cls)
            predicate_cls = int(predicate_cls)
            subj_id = int(subj_id)
            obj_id = int(obj_id)
            subj_label = self.object_class_names[int(subj_cls)]
            obj_label = self.object_class_names[int(obj_cls)]
            predicate_label = self.predicate_class_names[int(predicate_cls)]
            subj_center = np.stack(((subj_box[0] + subj_box[2]) / 2,
                                     (subj_box[1] + subj_box[3]) / 2),
                                    axis=0)
            obj_center = np.stack(((obj_box[0] + obj_box[2]) / 2,
                                     (obj_box[1] + obj_box[3]) / 2),
                                    axis=0)
            mid_center = (subj_center + obj_center) / 2

            # determine the location of the subject and object
            shift_x = (subj_center[0] < obj_center[0]) - 0.5
            shift_y = (subj_center[1] < obj_center[1]) - 0.5

            # then draw the label on the (predicate_cls + 1)-th output
            cur_vis = outputs[predicate_cls]
            objects_set = objects_sets[predicate_cls]
            if subj_id not in objects_set:
                self.draw_text_on_target(
                    cur_vis,
                    subj_label,
                    (subj_center[0]-20*shift_x, subj_center[1]-20*shift_y),
                    color=self.object_colormap(subj_cls),
                    horizontal_alignment="center",
                    font_size=self._default_font_size,
                )
                objects_set.add(subj_id)

            if obj_id not in objects_set:
                self.draw_text_on_target(
                    cur_vis,
                    obj_label,
                    (obj_center[0]+20*shift_x, obj_center[1]+20*shift_y),
                    color=self.object_colormap(obj_cls),
                    horizontal_alignment="center",
                    font_size=self._default_font_size,
                )
                objects_set.add(obj_id)

            # draw line
            self.draw_line_on_target(
                cur_vis,
                [subj_center[0], obj_center[0]],
                [subj_center[1], obj_center[1]],
                color="k",
                linewidth=0.1)
            # draw an arrow
            delta = obj_center - subj_center + 1e-12
            delta = 5 * delta / np.linalg.norm(delta)
            cur_vis.ax.arrow(mid_center[0], mid_center[1],
                             delta[0], delta[1],
                             head_width=6, head_length=5, fc="k",
                             ec="k",
                             )

            self.draw_text_on_target(
                cur_vis,
                "{:.0f}%".format(float(predicate_score * 100)),
                (mid_center[0] + 0, mid_center[1] + 10),
                color="white",
                horizontal_alignment="center",
                font_size=self._default_font_size,
            )
            # output_inds.add(predicate_cls)


        self.outputs = self.outputs + list(outputs.values())


        return self.outputs, len(existing_predicates)


    def draw_line_on_target(self, target, x_data, y_data, color, linestyle="--", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        target.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
                alpha=0.5
            )
        )
        return target

    def draw_text_on_target(
        self,
        target,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        alpha=0.5
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        target.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": alpha, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return target

# MIT License
# Credit: https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False, normalize=True):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    if normalize:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)