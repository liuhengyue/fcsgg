"""
Core modules for generating FCSGG ground-truth data.

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from fvcore.common.file_io import PathManager
from PIL import Image
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from fcsgg.structures.scene_graph import SceneGraph

__all__ = [
    "build_augmentation",
    "transform_instance_annotations",
    "annotations_to_scene_graph",
]

DEFAULT_FIELDS = ["gt_classes", "gt_ct_maps", "gt_wh", "gt_reg", "gt_centers_int",
    "gt_relations", "gt_relations_weights"]

SIZE_RANGE = {4: (0., 1/16), 8: (1/16, 1/8), 16: (1/8, 1/4), 32: (1/4, 1.)}

# SIZE_RANGE = {8: (0., 1/16), 16: (1/16, 1/8), 32: (1/8, 1/4), 64: (1/4, 1/2), 128: (1/2, 1.)}

dataset_meta = MetadataCatalog.get("vg_train")

def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.FLIP:
        augmentation.append(T.RandomFlip())
    return augmentation

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def annotations_to_instances(annos, relations, image_size):
    """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            relations (list[list]): (N, 3) triplet for N relations <s, o, predicate label>
            image_size (tuple): height, width

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    scene_graph = SceneGraph(image_size)
    gt_boxes = Boxes(boxes)

    gt_classes = [obj["category_id"] for obj in annos]
    gt_classes = torch.tensor(gt_classes, dtype=torch.int64)

    # we filter empty box here, as well as relations
    valid = gt_boxes.nonempty()
    gt_boxes = gt_boxes[valid]
    gt_classes = gt_classes[valid]
    if relations and (not valid.all()):
        valid_box_inds = valid.nonzero(as_tuple=True)[0]
        valid_box_inds = valid_box_inds.numpy()
        old2new = {ind: i for i, ind in enumerate(valid_box_inds)}
        filtered_relations = []
        for i in range(len(relations)):
            s_ind, o_ind, r = relations[i]
            if (s_ind not in valid_box_inds) or (o_ind not in valid_box_inds):
                continue
            # map to the new index
            filtered_relations.append([old2new[s_ind], old2new[o_ind], r])
        relations = filtered_relations
    # common fields
    scene_graph.set_extra("gt_boxes", gt_boxes)
    scene_graph.set_extra("gt_classes", gt_classes)
    relations = torch.as_tensor(relations, dtype=torch.long)
    # in case we got empty tensor
    if relations.numel() == 0:
        relations = relations.reshape((0, 3))
    scene_graph.set_extra("gt_relations", relations)
    return scene_graph



def get_oval_gaussian_radius(wh_tensor, min_overlap=0.7):
    """
    Return the two axis radius of the gaussian based on IOU min_overlap.
    Note this returns long tensors
    """
    factor = (1 - np.sqrt(min_overlap)) / np.sqrt(2) # > 0
    radius_a_b = wh_tensor * factor + 1
    return radius_a_b.long()

def gaussian2D(diameters, sigma_factor=6):
    num_instances = diameters.size(0)
    sigmas_x_y = diameters.float() / sigma_factor
    starts, ends = -diameters // 2, (diameters + 1) // 2
    guassian_masks = []
    # different gauss kernels have different range, had to use loop
    for i in range(num_instances):
        y, x = torch.meshgrid(torch.arange(starts[i][1], ends[i][1]),
                              torch.arange(starts[i][0], ends[i][0]))
        x = x.to(diameters.device)
        y = y.to(diameters.device)
        # range (0, 1]
        guassian_masks.append(torch.exp(-(x ** 2 / (2 * sigmas_x_y[i, 0] ** 2) +
                                          y ** 2 / (2 * sigmas_x_y[i, 1] ** 2))))
    return guassian_masks

"""

These two functions are just a walk-through of 
how to process each gaussian mask.

def draw_gaussian(fmap, center, radius, k=1):
    r_b, r_a = radius
    x, y = center[0], center[1]
    height, width = fmap.shape[:2]
    # r_b, r_a = int(r_b), int(r_a)
    # discretization
    left, right = torch.min(x, r_a), torch.min(width - x, r_a + 1)
    top, bottom = torch.min(y, r_b), torch.min(height - y, r_b + 1)
    y_top, y_bottom, x_left, x_right = (y - top).int(), (y + bottom).int(), \
                                       (x - left).int(), (x + right).int()
    masked_fmap  = fmap[y_top : y_bottom, x_left : x_right]
    # approximate guassian, we have to have gt = 1 in the heatmap
    diameter_a, diameter_b = x_right - x_left, y_bottom - y_top
    gaussian = gaussian2D((diameter_b, diameter_a))
    # masked_gaussian = gaussian[r_b - top:r_b + bottom, r_a - left:r_a + right]
    if min(gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
        masked_fmap = torch.max(masked_fmap, gaussian * k)
        fmap[y_top : y_bottom, x_left : x_right] = masked_fmap

def gaussian2D(diameter_b_a, sigma_factor=6):
    m, n = diameter_b_a #torch.floor(radius)
    sigma_b ,sigma_a = m.float() / sigma_factor, n.float() / sigma_factor
    y, x = torch.meshgrid(torch.arange(-m // 2, (m + 1) // 2), torch.arange(-n // 2, (n + 1) // 2))
    # range (0, 1]
    gauss = torch.exp(-(x ** 2 / (2 * sigma_a ** 2) + y ** 2 / (2 * sigma_b ** 2)))

"""






# now let's get the Relation Affinity Fields
def get_raf(gt_relations,
            gt_boxes,
            num_predicates,
            output_stride,
            output_size,
            sigma=0,
            range_wh=None,
            min_overlap=0.5,
            weight_by_length=False):
    """
    This function generates the relation affinity fields.
    gt_relations: Long tensor of shape (m, 3), note than m could be zero
    gt_centers: Float tensor of shape (n, 2) of (x, y)
    radius_a_b: Long tensor of shape (n, 2)
    m can be any number.

    Return : Rafs of shape: for VG-200, (50, 2, h, w) of (x, y)
    """

    device = gt_boxes.device
    # (P, 2, h, w)
    h, w = output_size
    rafs = torch.zeros((num_predicates, 2,) + output_size, device=device)
    # weight each relation equally or not, if no weight, then long and big relation field
    # will dominant the loss
    rafs_weights = torch.zeros((num_predicates, 2,) + output_size, device=device)

    # if no relations
    num_rels = gt_relations.size(0)
    if num_rels == 0:
        return rafs, rafs_weights

    gt_centers = gt_boxes.get_centers()  # in image scale
    # shape of (m, 2)
    # !important: objects can collapse into the same center
    subject_centers = gt_centers[gt_relations[:, 0]]
    object_centers = gt_centers[gt_relations[:, 1]]
    true_s2o_vectors = object_centers - subject_centers
    # if true vector collapse, we define vector [1e-6, 1e-6] for it
    true_s2o_vectors[true_s2o_vectors.eq(0).all(dim=1)] += 1e-6
    true_s2o_vectors_norms = torch.norm(true_s2o_vectors, dim=1, keepdim=False)
    # vector in feature

    subject_centers = subject_centers // output_stride
    object_centers = object_centers // output_stride
    s2o_vectors = object_centers - subject_centers
    # s2o_vectors = object_centers.floor() - subject_centers.floor()
    # if two centers are collapsed, we use true vector
    zero_vec_mask = s2o_vectors.eq(0).all(dim=1)
    s2o_vectors[zero_vec_mask] = true_s2o_vectors[zero_vec_mask]
    # right now it is a (1, 0) vector, not a random unit vector
    # rand_vec = torch.rand(2)
    # rand_vec /= rand_vec.norm()
    # s2o_vectors[zero_vec_mask] = torch.tensor([1.,0.], device=device)
    # shape (m, 1)
    s2o_vector_norms = torch.norm(s2o_vectors, dim=1, keepdim=False)

    if range_wh is not None:
        # check the vector norm, if it is in the range
        norm_range = torch.norm(range_wh, dim=0)
        valid_rel_mask = torch.logical_and(true_s2o_vectors_norms > norm_range[0],
                                           true_s2o_vectors_norms <= norm_range[1])
        # or either axis-aligned side
        # s2o_vectors_abs = torch.abs(s2o_vectors)
        # valid_rel_mask = torch.logical_and(s2o_vectors_abs > range_wh[..., 0],
        #                                    s2o_vectors_abs <= range_wh[..., 1]).any(dim=-1)
        gt_relations = gt_relations[valid_rel_mask]
        s2o_vectors = s2o_vectors[valid_rel_mask]
        s2o_vector_norms = s2o_vector_norms[valid_rel_mask][..., None]
        subject_centers = subject_centers[valid_rel_mask]
        object_centers = object_centers[valid_rel_mask]
        num_rels = gt_relations.size(0)
        # for some scales, there could be no relations
        if num_rels == 0:
            return rafs, rafs_weights
    else:
        s2o_vector_norms = s2o_vector_norms[..., None]


    # count the number of raf overlap at each pixel location
    cross_raf_counts = torch.zeros((num_predicates,) + output_size, device=device)

    if sigma == 0:
        gt_boxes.scale(1 / output_stride, 1 / output_stride)
        gt_wh = torch.stack((gt_boxes.tensor[..., 2] - gt_boxes.tensor[..., 0],  # width in downsampled feat-level
                             gt_boxes.tensor[..., 3] - gt_boxes.tensor[..., 1]),  # height in downsampled feat-level
                    dim=-1)
        radius_a_b = get_oval_gaussian_radius(gt_wh, min_overlap=min_overlap)
        # raf width dependent on the radius r_s, r_o
        subject_radius = radius_a_b[gt_relations[:, 0]]
        object_radius = radius_a_b[gt_relations[:, 1]]
        sigma = torch.cat((subject_radius, object_radius), dim=1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    else:
        # sigma = torch.ones((num_rels, 1, 1), device=device) * sigma
        sigma = torch.ones((num_rels, 1, 1), device=device) * np.sqrt(128 / output_stride)
    relation_unit_vecs = s2o_vectors / s2o_vector_norms # shape (m, 2)
    # for assign weights, longer relation has smaller weights
    relation_distance_weights = F.normalize(1 / s2o_vector_norms, p=2, dim=0)
    # shape (m, w, h)
    m, y, x = torch.meshgrid(torch.arange(num_rels, device=device),
                             torch.arange(h, device=device),
                             torch.arange(w, device=device))
    # equation: v = (v_x, v_y) unit vector along s->o, a center point c = (c_x, c_y), any point p = (x, y)
    # <v, p - c> = v_x * (x - c_x) + v_y * (y - c_y) gives distance along the relation vector
    # <v⊥, p - c> gives distance orthogonal to the relation vector
    dist_along_rel = torch.abs(relation_unit_vecs[:, 0:1, None] * (x - (subject_centers[:, 0:1, None] + object_centers[:, 0:1, None]) / 2) + \
                      relation_unit_vecs[:, 1:2, None] * (y - (subject_centers[:, 1:2, None] + object_centers[:, 1:2, None]) / 2))
    dist_ortho_rel = torch.abs(relation_unit_vecs[:, 1:2, None] * (x - subject_centers[:, 0:1, None]) - \
                      relation_unit_vecs[:, 0:1, None] * (y - subject_centers[:, 1:2, None]))
    # valid = (dist_along_rel >= 0) \
    #         * (dist_along_rel <= s2o_vector_norms[..., None]) \
    #         * (dist_ortho_rel <= sigma)
    valid = (dist_along_rel <= s2o_vector_norms[..., None] / 2 + sigma // 2) \
            * (dist_ortho_rel <= sigma)
    # (m, w, h) <-- (m, 2) (m, w, h)
    rafs_x = relation_unit_vecs[:, 0:1, None] * valid
    rafs_y = relation_unit_vecs[:, 1:2, None] * valid
    valid = valid.float() # for computing the weights
    rafs_weights_ortho_rel = torch.min(torch.exp(-torch.clamp(dist_along_rel - s2o_vector_norms[..., None] / 2,\
                                                              min=0).round() / 1),\
                             torch.exp(-torch.round(dist_ortho_rel) / 1))  * valid # [0, 1]
    # rafs_weights = rafs_weights_ortho_rel.max(dim=0).values
    # gather by predicate class (not sure it is fast enough) to shape (50, 2, h, w)
    gt_predicates = gt_relations[:,2]
    for i, gt_predicate in enumerate(gt_predicates):
        cross_raf_counts[gt_predicate, ...] += torch.logical_or(rafs_x[i], rafs_y[i])
        rafs[gt_predicate, 0] += rafs_x[i]
        rafs[gt_predicate, 1] += rafs_y[i]
        # OPTIONAL: if overlapped, the weight is higher too; normalize by relation area/length
        # rafs_weights_ortho_rel[i].eq(1).sum()
        if weight_by_length:
            rafs_weights[gt_predicate] = torch.max(rafs_weights[gt_predicate],
                                                   rafs_weights_ortho_rel[i] * relation_distance_weights[i])
        else:
            rafs_weights[gt_predicate] = torch.max(rafs_weights[gt_predicate], rafs_weights_ortho_rel[i])
    # divide by number of intersections
    rafs = torch.where(cross_raf_counts[:, None, ...] > 1, rafs / cross_raf_counts[:, None, ...], rafs)
    # for weights, if there is one object with many relations, then its weight should be low.
    # Why? At the object location, the raf will have many vectors superposition, it is impossible to
    # predict such vector
    counts_sum = cross_raf_counts[:, None, :, :]
    rafs_weights = torch.where(counts_sum > 1,
                               rafs_weights / counts_sum, rafs_weights)

    return rafs, rafs_weights

class GroundTruthGen(object):
    """
        A class for generating the ground-truth on the fly given the configurations.
        It will be passed as the instance variable of the meta architecture (CenterNet).

    """

    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.output_strides = cfg.MODEL.HEADS.OUTPUT_STRIDES
        self.gt_ious = cfg.INPUT.CENTER_GAUSSIAN_IOU_MIN_OVERLAPS
        assert len(self.output_strides) == len(self.gt_ious), "for each stride, there should be one iou threshold."
        self.split_by_scale = cfg.INPUT.GT_SCALE_AWARE
        self.regression_split_by_scale = cfg.INPUT.GT_REGRESSION_SCALE_AWARE
        self.raf_stride_ratio = cfg.INPUT.RAF_STRIDE_RATIO
        if len(self.output_strides) == 3:
            self.size_range = [(1e-12, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1.)]
        elif len(self.output_strides) == 4:
            self.size_range = [(1e-12, 1 / 16),
                               (1 / 16, 1 / 8),
                               (1 / 8, 1 / 4),
                               (1 / 4, 1.)]
        elif len(self.output_strides) == 5:
            self.size_range = [(1e-12, 1 / 16),
                               (1 / 16, 1 / 8),
                               (1 / 8, 1 / 4),
                               (1 / 4, 1 / 2),
                               (1 / 2, 1.)]
        else:
            self.size_range = [None]
        self.max_size = cfg.INPUT.MAX_SIZE_TRAIN
        self.use_fix_size = cfg.INPUT.USE_FIX_SIZE

        # relation related fields
        self.load_relations = cfg.RELATION.RELATION_ON
        self.raf_type = cfg.INPUT.RAF_TYPE

    def generate_score_map(self,
                           num_classes,
                           gt_classes,
                           gt_centers,
                           radius_a_b,
                           output_size,
                           padding,
                           device,
                           k=1.):
        fmap = torch.zeros((num_classes,) + output_size, device=device)
        _, height, width = fmap.shape
        wh_tensor = torch.tensor([width, height], device=gt_centers.device)
        # discretization
        left_top = torch.min(gt_centers, radius_a_b)
        right_bottom = torch.min(wh_tensor - gt_centers, radius_a_b + 1)
        mask_left_top = (gt_centers - left_top).int()
        mask_right_bottom = (gt_centers + right_bottom).int()
        # approximate guassian, we have to have gt = 1 in the heatmap
        diameters = mask_right_bottom - mask_left_top  # order x, y
        gaussian_masks = gaussian2D(diameters)
        for i, gaussian_mask in enumerate(gaussian_masks):
            masked_fmap = fmap[gt_classes[i],
                          mask_left_top[i, 1]: mask_right_bottom[i, 1],
                          mask_left_top[i, 0]: mask_right_bottom[i, 0]
                          ]
            masked_fmap = torch.max(masked_fmap, gaussian_mask * k)
            fmap[gt_classes[i],
            mask_left_top[i, 1]: mask_right_bottom[i, 1],
            mask_left_top[i, 0]: mask_right_bottom[i, 0]] = masked_fmap
        fmap = F.pad(fmap, padding, value=0)
        return fmap


    def sort_box_in_range(self,
                          gt_boxes: torch.Tensor,
                          gt_classes: torch.Tensor,
                          range_wh: torch.Tensor,
                          image_size: Tuple[int, int],
                          output_stride: int):
        # Based on region area
        gt_area = gt_boxes.area()
        range_area = range_wh.prod(dim=0).to(device=gt_area.device)
        valid = torch.logical_and(gt_area > range_area[0], gt_area <= range_area[1])
        # based on either side
        # valid = torch.logical_and(gt_wh >= range_wh[..., 0], gt_wh <= range_wh[..., 1]).any(dim=-1)
        gt_boxes = gt_boxes[valid]
        gt_classes = gt_classes[valid]
        gt_area = gt_area[valid].cpu().numpy()
        # then sort by area
        gt_centers = gt_boxes.get_centers().cpu().numpy() // output_stride
        bbox_center1d = gt_centers[..., 1] * image_size[1] // output_stride + gt_centers[..., 0]
        # sort by center first then area
        sorted_inds = np.lexsort((gt_area, bbox_center1d))
        return gt_boxes[sorted_inds], gt_classes[sorted_inds]

    # now let's get the Relation Affinity Fields
    def get_raf(self,
                num_predicates,
                gt_relations,
                gt_centers,
                radius_a_b,
                output_size,
                padding_size):
        """
        This function generates the relation affinity fields in the form of points.
        gt_relations: Long tensor of shape (m, 3), note than m could be zero
        gt_centers: Float tensor of shape (n, 2) of (x, y)
        radius_a_b: Long tensor of shape (n, 2)
        m can be any number.

        Return : Rafs of shape: for VG-200, (50, 2, h, w) of (x, y)
        """

        device = gt_centers.device

        # if no relations
        num_rels = gt_relations.size(0)
        if num_rels == 0:
            gt_rafs = torch.zeros((num_predicates, 2,) + output_size, device=device)
            gt_rafs_weights = torch.zeros((num_predicates, 2,) + output_size, device=device)
            gt_rafs = F.pad(gt_rafs, padding_size, value=0)
            gt_rafs_weights = F.pad(gt_rafs_weights, padding_size, value=0)
            return gt_rafs, gt_rafs_weights

        # rafs dim 2, 3
        subject_centers = gt_centers[gt_relations[:, 0]]
        subject_radius = radius_a_b[gt_relations[:, 0]]
        object_centers = gt_centers[gt_relations[:, 1]]
        object_radius = radius_a_b[gt_relations[:, 1]]
        subj_maps = self.generate_score_map(num_predicates,
                                             gt_relations[:, 2],
                                             subject_centers,
                                             subject_radius,
                                             output_size, padding_size, device)

        obj_maps = self.generate_score_map(num_predicates,
                                            gt_relations[:, 2],
                                            object_centers,
                                            object_radius,
                                            output_size, padding_size, device)

        rafs = torch.stack((subj_maps, obj_maps), dim=1)
        rafs_weights = torch.ones_like(rafs)


        return rafs, rafs_weights


    def generate_gt_scale(self, sg, target_image_size, output_stride, iou_threshold,
                          size_range, training=True):
        """
        image_size: the image size used during training after padding
        sg.image_size will be the image size after augmentation and divisible to a fixed number.
        """
        image_size = sg.image_size
        target_output_size = tuple(np.floor_divide(target_image_size, output_stride))
        # shape (H/4, W/4) ... (H/32, W/32)
        output_size = tuple(np.floor_divide(image_size, output_stride))
        # [w left, w right, h top, h bottom]
        padding_size = [0, target_output_size[1] - output_size[1],
                        0, target_output_size[0] - output_size[0]]  # right and bottom
        gt_boxes = sg.get_extra("gt_boxes").clone()
        gt_classes = sg.get_extra("gt_classes").clone()
        gt_relations = sg.get_extra("gt_relations").clone()
        device = gt_boxes.device
        range_wh = None

        if self.split_by_scale:
            if self.use_fix_size:
                image_size_tensor = torch.tensor((self.max_size, self.max_size), dtype=torch.float)
            else:
                image_size_tensor = torch.tensor(image_size, dtype=torch.float)
            range_side = torch.tensor(size_range)
            # range in image scale
            range_wh = image_size_tensor.unsqueeze(1) @ range_side.unsqueeze(0)
            gt_boxes, gt_classes = self.sort_box_in_range(gt_boxes, gt_classes,
                                                          range_wh, image_size, output_stride)

        # shape (num_instances, 2 {x, y})
        gt_boxes.scale(1 / output_stride, 1 / output_stride)
        gt_centers = gt_boxes.get_centers()
        gt_centers_int = gt_centers.to(torch.long)  # collect

        # centers are (x, y), output_size[1] is the width
        # We use the padded image size, so the index can be generated here
        gt_index = gt_centers_int[..., 1] * target_output_size[1] + gt_centers_int[..., 0] # collect
        gt_reg = gt_centers - gt_centers_int  # collect
        # xyxy, gt_wh is float number in range (0, feature size)
        gt_wh = torch.stack((gt_boxes.tensor[..., 2] - gt_boxes.tensor[..., 0],  # width in downsampled feat-level
                             gt_boxes.tensor[..., 3] - gt_boxes.tensor[..., 1]), # height in downsampled feat-level
                            dim=-1)

        radius_a_b = get_oval_gaussian_radius(gt_wh, min_overlap=iou_threshold)

        if self.split_by_scale and training:
            # the bboxes are already sorted, the first occurrence of the element will be of smallest area
            _, inds = np.unique(gt_index.cpu().numpy(), return_index=True)
            # now only keep unique regression targets
            gt_index = gt_index[inds]
            gt_reg = gt_reg[inds]
            gt_wh = gt_wh[inds]
        # prepare the gt dict then the following functions can modify in-place

        gt_dict = {
            "gt_classes": gt_classes,
            "gt_index": gt_index,
            # these three are needed for regression and PredCls #
            "gt_wh": gt_wh,
            "gt_reg": gt_reg,
            "gt_centers_int": gt_centers_int,
            # for center and raf gt generation
            # "radius_a_b": radius_a_b
        }
        # modify gt_ct_maps
        ct_ht_maps = self.generate_score_map(self.num_classes,
                                            gt_classes,
                                            gt_centers_int,
                                            radius_a_b,
                                            output_size, padding_size, device)
        gt_dict.update({"gt_ct_maps": ct_ht_maps})

        # this confirms every point on the location should be 1
        # assert gt_ct_maps[gt_classes, gt_centers_int[:, 1], gt_centers_int[:, 0]].mean() == 1, "wrong heatmap generation."
        gt_rafs, gt_rafs_weights = None, None
        gt_num_relations = 0
        if self.load_relations:
            gt_num_relations = gt_relations.size(0)
            if self.raf_stride_ratio != 1:
                # gt_centers_int = (gt_centers // self.raf_stride_ratio).to(torch.long)
                radius_a_b = get_oval_gaussian_radius(gt_wh / 2, min_overlap=iou_threshold)
                output_size = tuple(np.floor_divide(image_size, output_stride * self.raf_stride_ratio))

            ###############  approach 1 ##############
            if self.raf_type == "vector":
                # now takes in gt_boxes
                gt_boxes = sg.get_extra("gt_boxes").clone()
                gt_rafs, gt_rafs_weights = get_raf(gt_relations, gt_boxes,
                                                   self.num_predicates,
                                                   output_stride,
                                                   output_size,
                                                   range_wh=range_wh,
                                                   min_overlap=iou_threshold)

                # important! pad gt_rafs, gt_rafs_weights
                gt_rafs = F.pad(gt_rafs, padding_size, value=0)
                gt_rafs_weights = F.pad(gt_rafs_weights, padding_size, value=0)

            ###############  approach 2 ##############
            # This approach does not work well.
            elif self.raf_type == "point":
                gt_rafs, gt_rafs_weights = self.get_raf(self.num_predicates,
                                                       gt_relations,
                                                       gt_centers_int,
                                                       radius_a_b,
                                                       output_size,
                                                       padding_size=padding_size)
            else:
                raise NotImplementedError()

        gt_dict.update({"gt_relations": gt_rafs,
                        "gt_relations_weights": gt_rafs_weights,
                        "gt_num_relations": gt_num_relations})

        return gt_dict

    def __call__(self, sg, image_size, image_id=0, training=True):
        sg.init()
        for gt_iou, out_stride, size_range in zip(self.gt_ious, self.output_strides, self.size_range):
            gt_dict = self.generate_gt_scale(sg, image_size,
                                             out_stride, gt_iou,
                                             size_range,
                                             training=training)
            sg.update(gt_dict)
        return sg