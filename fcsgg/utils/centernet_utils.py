"""
Core implementations of decoding heatmaps into bounding box detections for CenterNet.
Modified from https://github.com/FateScript/CenterNet-better.
Some loss functions are not used, but keep it here for potential usage.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["wangfeng19950315@163.com https://github.com/FateScript/CenterNet-better"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import math
import functools
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from detectron2.structures import Instances, Boxes
from detectron2.layers import batched_nms
from detectron2.data import MetadataCatalog
from fcsgg.structures import SceneGraph
from typing import List, Tuple, Union, Dict, Optional
import logging


__all__ = ["reg_smooth_l1_loss", "reg_l1_loss", "focal_loss",
           "efficient_focal_loss", "mse_loss", "gather_feature",
           "raf_loss", "reg_log_l1_loss"]



def reg_l1_loss(output, index, target):
    """
    index: shape of (2, N) in order of batch_idx, 1d pixel location
    target: shape of (N, 2)
    """
    # we may have empty gt
    if target.size(0):
        batch, channel = output.shape[:2]
        output = output.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        output = output[index[0], index[1], :]
        loss = F.l1_loss(output, target, reduction='mean')
    else:
        loss = 0.0 * output.sum()
    return loss

def reg_log_l1_loss(output, index, target):
    # we may have empty gt
    if target.size(0):
        batch, channel = output.shape[:2]
        output = output.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        output = output[index[0], index[1], :]
        loss = F.l1_loss(torch.log(output / target + 1e-16), torch.zeros_like(target), reduction='mean')
    else:
        loss = 0.0 * output.sum()
    return loss


def reg_smooth_l1_loss(output, index, target):
    if target.size(0):
        batch, channel = output.shape[:2]
        output = output.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        output = output[index[0], index[2], :]
        loss = F.smooth_l1_loss(output, target, reduction='mean')
    else:
        loss = 0.0 * output.sum()
    return loss

def focal_loss(pred, targets, alpha=2, beta=4, index=None):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    gt = torch.stack([x.gt_ct_maps for x in targets], dim=0)

    pos_inds = gt.eq(1).float() # No. objects in image
    # neg_inds = ((gt < 1) * (gt > 0)).float()
    neg_inds = gt.lt(1).float() #

    neg_weights = torch.pow(1 - gt, beta)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, min=1e-12, max=1-1e-12)

    # pos_loss is larger than neg_loss
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def efficient_focal_loss(pred, targets):
    '''
    Arguments:
      pred, targets: B x C x H x W
      index: 3 x M
    '''
    gt_heatmaps = torch.stack([x.gt_ct_maps for x in targets], dim=0)

    return gaussian_focal_loss(pred, gt_heatmaps)

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    From mmdetection:
    https://github.com/open-mmlab/mmdetection/blob/ced1c57dd212a8542f5a6fbc504eb22338a49e65/mmdet/models/losses/gaussian_focal_loss.py
    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    pos_num = pos_weights.sum().clamp(min=1)
    return (pos_loss + neg_loss).sum() / pos_num


def mse_loss(pred, targets, index=None):
    gt = torch.stack([x.gt_ct_maps for x in targets], dim=0)
    gt_mask = gt.gt(0)
    num_gt = gt.eq(1).sum()
    return ((gt_mask * (pred - gt)) ** 2).sum() / num_gt


def raf_loss(rafs, targets, mask=True):
    gt_rafs = torch.stack([x.gt_relations for x in targets], dim=0)
    rafs = rafs.view_as(gt_rafs)
    gt_raf_weights = torch.stack([x.gt_relations_weights for x in targets], dim=0)
    num_valid = gt_raf_weights.eq(1).sum().clamp(min=1)
    # batched_num_valid = gt_raf_weights.ge(1).sum(dim=[1, 2]) # shape (b, )
    # in case we have no relation, the loss will still be zero since gt_raf_weights are all zero
    # batched_num_valid.clamp_(min=1)
    # gt_raf_weights.unsqueeze_(1)
    angle = torch.pow(-F.cosine_similarity(rafs, gt_rafs, dim=2) + 1.1, 2) # [1, 4]
    loss = (F.mse_loss(rafs, gt_rafs, reduction='none') * gt_raf_weights).sum(dim=2)
    loss = (loss * angle).sum() / num_valid
    # loss = torch.mean(loss / batched_num_valid)
    return loss

def quantile_loss(input, target, q=0.25):
    e = input - target
    loss = torch.max(q * e, (q - 1) * e)
    return loss

def raf_focal_loss(rafs, targets, alpha=1):
    gt_rafs = torch.stack([x.gt_relations for x in targets], dim=0)
    b, p, _, h, w = gt_rafs.size()
    # gt_rafs = gt_rafs.view(b, -1, h, w)
    rafs = rafs.view_as(gt_rafs)
    gt_raf_weights = torch.stack([x.gt_relations_weights for x in targets], dim=0)
    batched_num_valid = gt_raf_weights.eq(1).sum(dim=[1, 2]) # shape (b, )
    # in case we have no relation, the loss will still be zero since gt_raf_weights are all zero
    batched_num_valid.clamp_(min=1)
    # gt_raf_weights.unsqueeze_(1) # (b, 1, h, w)
    # gt_raf_weights.unsqueeze_(1)  # (b, 1, h, w)
    # loss = (F.mse_loss(rafs, gt_rafs, reduction='none') * gt_raf_weights).sum(dim=[1, 2, 3])
    # loss = torch.mean(loss / batched_num_valid)
    p = (1 - F.cosine_similarity(rafs, gt_rafs, dim=2)) / 2 # range [0, 1]
    dis = F.mse_loss(rafs, gt_rafs, reduction='none') * gt_raf_weights.unsqueeze(1).unsqueeze(1)
    # neg_inds = torch.logical_and(gt_raf_weights.lt(1), gt_raf_weights.gt(0)).nonzero(as_tuple=True)
    neg_inds = gt_raf_weights.lt(1).nonzero(as_tuple=True)
    pos_inds = gt_raf_weights.eq(1).nonzero(as_tuple=True)
    # negative loss
    # neg_loss = p[neg_inds[0], : , None, neg_inds[2], neg_inds[3]].pow(alpha) * \
    #            dis[neg_inds[0], : , :, neg_inds[2], neg_inds[3]].pow(4)
    # neg_loss = neg_loss.sum()
    neg_loss = 0 * rafs.sum()
    # positive loss
    num_pos = pos_inds[0].size(0)
    # pos_loss = (1 - p[pos_inds[0], : , None, pos_inds[2], pos_inds[3]]).pow(alpha) * \
    #            dis[pos_inds[0], : , :, pos_inds[2], pos_inds[3]]
    pos_loss = dis[pos_inds[0], : , :, pos_inds[1], pos_inds[2]]
    pos_loss = pos_loss.sum()
    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos





def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CenterNetDecoder(object):

    def __init__(self, relation_on, use_gt_box, use_gt_object_label,
                 output_strides=[4], raf_stride_ratio=1, raf_type="vector",
                 use_fix_size=False, max_size_test=1024,
                 topk_per_image=100, nms_thresh=0.7, freq_bias=False,
                 relation_only=False):
        self.topk_per_image = topk_per_image
        self.nms_thresh = nms_thresh
        self.output_strides = output_strides
        self.max_stride = output_strides[-1]
        self.raf_stride_ratio = raf_stride_ratio
        self.relation_on = relation_on
        self.use_gt_box = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.raf_type = raf_type
        self.relation_only = relation_only
        self.device = "cpu" # this will be changed in `decode`
        # self.SIZE_RANGE = {4: (1e-12, 1/16), 8: (1/16, 1/8), 16: (1/8, 1/4), 32: (1/4, 1.)} if len(output_strides) == 4 \
        #     else {8: (0, 1/16), 16: (1/16, 1/8), 32: (1/8, 1/4), 64: (1/4, 1/2), 128: (1/2, 1.)}
        eps = 0.0
        if len(output_strides) == 3:
            self.SIZE_RANGE = [(1e-12, (1 + eps) / 8),
                               ((1 - eps) / 8, (1 + eps) / 4),
                               ((1 - eps) / 4, 1.)]
        elif len(output_strides) == 4:
            self.SIZE_RANGE = [(1e-12, (1 + eps) / 16),
                               ((1 - eps) / 16, (1 + eps) / 8),
                               ((1 - eps) / 8, (1 + eps) / 4),
                               ((1 - eps) / 4, 1.)]
        elif len(output_strides) == 5:
            self.SIZE_RANGE = [(1e-12, (1 + eps) / 16),
                               ((1 - eps) / 16, (1 + eps) / 8),
                               ((1 - eps) / 8, (1 + eps) / 4),
                               ((1 - eps) / 4, (1 + eps) / 2),
                               ((1 - eps) / 2, 1.)]
        self.max_size = max_size_test
        self.use_fix_size = use_fix_size
        if freq_bias:
            self.freq_bias = torch.as_tensor(MetadataCatalog.get("vg_train").get("frequency_bias"))
            self.freq_bias = 1.0001 ** self.freq_bias
        else:
            self.freq_bias = None

        self._logger = logging.getLogger("detectron2.evaluation.evaluator")


    def decode_multiscale_results_slow(self, results, top_k=100, nms_thresh=0.5):
        """
        When we sort all levels together, it is 2x slower than do top k for each scale first
        """
        fmaps = [result.get("cls", None).detach() for result in results]
        wh_maps = [result.get("wh", None).detach() for result in results]
        reg_maps = [result.get("reg", None).detach() for result in results]
        device = fmaps[0].device
        has_reg = None not in reg_maps
        fmap_shapes = [fmap.size() for fmap in fmaps]
        # record for each match, the corresponding scale
        fmaps_flattened, scale_inds = [], []
        # accumulated number of pixels for each scale
        accumulated_sum = [0]

        for i, (fmap, stride) in enumerate(zip(fmaps, self.output_strides)):
            batch, channel, height, width = fmap_shapes[i]
            fmap = self.pseudo_nms(fmap)
            fmap = fmap.reshape(batch, -1)
            numels = fmap.size(-1)
            this_scale_inds = torch.zeros(numels, device=device, dtype=torch.long)
            this_scale_inds.fill_(i)
            scale_inds.append(this_scale_inds)
            fmaps_flattened.append(fmap)
            accumulated_sum.append(numels)
        accumulated_sum = torch.as_tensor(np.cumsum(accumulated_sum), device=device)
        scale_inds = torch.cat(scale_inds)
        fmaps_flattened = torch.cat(fmaps_flattened, dim=-1)
        scores, inds = torch.topk(fmaps_flattened, top_k)
        scale_inds = scale_inds[inds]
        inds = inds - accumulated_sum[scale_inds]
        inds = inds.cpu()
        detections = []
        # return detections
        for b in range(inds.size(0)):
            per_img_detections = []
            for i in range(inds.size(1)):
                scale_ind = scale_inds[b, i]
                ind = inds[b, i]
                score = scores[b, i]
                fmap_shape = fmap_shapes[scale_ind][1:]
                fmap_inds = np.unravel_index(ind, fmap_shape)
                # get other detections
                wh = wh_maps[scale_ind][b, :, fmap_inds[1], fmap_inds[2]]
                if has_reg:
                    reg = reg_maps[scale_ind][b, :, fmap_inds[1], fmap_inds[2]]
                else:
                    reg = torch.tensor((0., 0.))
                detection = torch.cat((torch.as_tensor(fmap_inds, device=device),
                                       wh,
                                       reg,
                                       torch.as_tensor([self.output_strides[scale_ind]], device=device),
                                       torch.as_tensor([score], device=device)))
                per_img_detections.append(detection)
            detections.append(torch.stack(per_img_detections))
        # shape (B, K, 9), each triplet is the class label, y, x, w, h, reg_x, reg_y, stride, score
        detections = torch.stack(detections)
        # we map the detection back to the image level
        pred_classes = detections[..., 0]
        pred_scores = detections[..., 8]
        centers_y = (detections[..., 1] + detections[..., 6]) * detections[..., 7]
        centers_x = (detections[..., 2] + detections[..., 5]) * detections[..., 7]
        half_w, half_h = detections[..., 7] * detections[..., 3] / 2, detections[..., 7] * detections[..., 4] / 2
        pred_bboxes = torch.stack([centers_x - half_w, centers_y - half_h,
                            centers_x + half_w, centers_y + half_h],
                           dim=-1)
        detections_as_list = []
        for i in range(detections.size(0)):
            detection_per_image = {}
            detection_per_image["pred_boxes"] = pred_bboxes[i]
            detection_per_image["scores"] = pred_scores[i]
            detection_per_image["pred_classes"] = pred_classes[i]
            detections_as_list.append(detection_per_image)
        return detections_as_list

    def decode_multiscale_results(self, results, batched_inputs):
        detections_all_levels = [defaultdict(list) for _ in range(len(batched_inputs))]
        for i, out_stride in enumerate(self.output_strides):
            detections = self.decode_detections(results[i], output_stride=out_stride, nms=False)
            for b, det in enumerate(detections):
                for k, v in det.items():
                    detections_all_levels[b][k].append(v)
        # merge results from all strides
        for i in range(len(detections_all_levels)):
            for k, v in detections_all_levels[i].items():
                detections_all_levels[i][k] = torch.cat(v, dim=0)
            # also change to dict since it is no long a list
            detections_all_levels[i] = dict(detections_all_levels[i])
        # nms
        for i in range(len(detections_all_levels)):
            det = detections_all_levels[i]
            boxes = det['pred_boxes']
            scores = det['scores']
            clses = det['pred_classes']
            centers = det['pred_centers']
            # _, keep = torch.topk(scores, self.topk_per_image)
            # returned the index of the kept boxes sorted by scores
            keep = batched_nms(boxes, scores, clses, self.nms_thresh)
            if self.topk_per_image >= 0:
                keep = keep[:self.topk_per_image]
            boxes, scores, clses, centers = boxes[keep], scores[keep], clses[keep], centers[keep]
            detections_all_levels[i]['pred_boxes'] = boxes
            detections_all_levels[i]['scores'] = scores
            detections_all_levels[i]['pred_classes'] = clses
            detections_all_levels[i]['pred_centers'] = centers
        return detections_all_levels

    def decode(self, results, batched_inputs, image_sizes, device="cpu", nms_thresh=0.5):
        r"""
        decode output feature map to detection results

        Args:
            results (batch of images) contains following fields:
                fmap(Tensor): output feature map
                wh(Tensor): tensor that represents predicted width-height
                reg(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `wh` or not
            K(int): topk value
        """
        self.device = device
        if not self.relation_only:
            if isinstance(results, list):
                detections = self.decode_multiscale_results(results, batched_inputs)
            else:
                output_stride = self.output_strides[0]
                detections = self.decode_detections(results, output_stride=output_stride, nms=True)
        else:
            detections = [{} for _ in range(len(batched_inputs))]
        # modify detection results
        self.modify_detection_results(results, detections, batched_inputs, debug=False)
        if self.relation_on:
            if isinstance(results, list):
                # this will modify detections
                self.decode_multiscale_relations(results, detections, image_sizes)
            else:
                if self.raf_type == "vector":
                    self.decode_relations(results, detections, output_stride=output_stride)
                elif self.raf_type == "point":
                    self.decode_point_relations(results, detections, output_stride=output_stride)
                else:
                    raise NotImplementedError()
        return detections


    def modify_detection_results(self, results, detections, batched_inputs, debug=False):
        if self.use_gt_box:
            num_images = len(batched_inputs)
            batched_gt_centers_int = [input["scene_graph"].gt_centers_int for input in batched_inputs]
            batched_gt_center_maps = [input["scene_graph"].gt_ct_maps for input in batched_inputs]
            batched_gt_whs = [input["scene_graph"].gt_wh for input in batched_inputs]
            batched_gt_regs = [input["scene_graph"].gt_reg for input in batched_inputs]
            if isinstance(results, dict):
               results = [results]
            fmaps = [result.get("cls", None) for result in results]
            # fmaps = [gt_center_maps.unsqueeze(0) for gt_center_maps in batched_gt_center_maps[0]]
            pred_whs = [result.get("wh", None) for result in results][0]
            pred_regs = [result.get("reg", None) for result in results][0]
            for i in range(num_images):
                pred_centers = []
                scores = []
                pred_classes = []
                pred_boxes = []
                # pred_centers = [torch.empty((0, 2), device=self.device)]
                # scores = [torch.empty((0), device=self.device)]
                # pred_classes = [torch.empty((0), device=self.device)]
                # pred_boxes = [torch.empty((0, 4), device=self.device)]
                gt_centers_int = batched_gt_centers_int[i]
                gt_whs = batched_gt_whs[i]
                gt_regs = batched_gt_regs[i]
                for j, (fmap, gt_cts, gt_wh, gt_reg, stride) in enumerate(zip(fmaps, gt_centers_int, gt_whs, gt_regs, self.output_strides)):
                    if gt_cts.size(0):
                        scores_per_scale, clses_per_scale = fmap[i, :, gt_cts[:, 1], gt_cts[:, 0]].max(dim=0)
                        # we can debug pred here
                        # pred_wh_per_scale = torch.transpose(pred_whs[i, :, gt_cts[:, 1], gt_cts[:, 0]], 1, 0) * stride
                        # pred_reg_per_scale = torch.transpose(pred_regs[i, :, gt_cts[:, 1], gt_cts[:, 0]], 1, 0)
                        gt_wh_per_scale = gt_wh * stride
                        gt_reg_per_scale = gt_reg
                        # self._logger.info(pred_reg_per_scale, gt_reg_per_scale)
                        gt_cts_per_scale = (gt_cts + gt_reg_per_scale) * stride

                        half_w, half_h = gt_wh_per_scale[..., 0] / 2, gt_wh_per_scale[..., 1] / 2
                        bboxes_per_scale = torch.stack([gt_cts_per_scale[:, 0] - half_w, gt_cts_per_scale[:, 1] - half_h,
                                            gt_cts_per_scale[:, 0] + half_w, gt_cts_per_scale[:, 1] + half_h],
                                           dim=-1)
                        pred_boxes.append(bboxes_per_scale)
                        scores.append(scores_per_scale)
                        pred_classes.append(clses_per_scale)
                        pred_centers.append(gt_cts_per_scale)
                detections[i]["pred_centers"] = torch.cat(pred_centers)
                detections[i]["scores"] = torch.cat(scores)
                detections[i]["pred_classes"] = torch.cat(pred_classes)
                detections[i]["pred_boxes"] = torch.cat(pred_boxes)

        elif self.use_gt_object_label:
            num_images = len(batched_inputs)
            for i in range(num_images):
                gt_boxes = batched_inputs[i]["scene_graph"].get_extra("gt_boxes")
                detections[i]["pred_boxes"] = gt_boxes.tensor
                detections[i]["pred_classes"] = batched_inputs[i]["scene_graph"].get_extra("gt_classes")
                detections[i]["scores"] = torch.ones_like(detections[i]["pred_classes"], dtype=torch.float)
                detections[i]["pred_centers"] = gt_boxes.get_centers()

        # Even if with GT, the evaluation would not be `perfect` since there are relations with same s-o pairs
        if debug:
            gt_rafs = [input["scene_graph"].gt_relations for input in batched_inputs]
            if isinstance(results, list):
                for stride, result in enumerate(results):
                    gt_rafs_per_scale = torch.stack([gt_rafs[i][stride] for i in range(len(gt_rafs))], dim=0)
                    gt_rafs_per_scale = gt_rafs_per_scale.view(gt_rafs_per_scale.size(0), -1, gt_rafs_per_scale.size(3), gt_rafs_per_scale.size(4))

                    results[stride]['raf'] = gt_rafs_per_scale # torch.tanh(gt_rafs_per_scale)
            else:
                gt_rafs = torch.stack([gt_rafs[i][0] for i in range(len(gt_rafs))], dim=0)
                gt_rafs = gt_rafs.view(gt_rafs.size(0), -1, gt_rafs.size(3), gt_rafs.size(4))
                results['raf'] = gt_rafs

    def decode_detections_from_wh(self, results, output_stride=4, nms=False,
                          cat_spec_wh=False, K=100, threshold=0.0, wh_range=True):
        # scores = self.get_scores(results)
        fmap = results.get("cls", None)
        reg = results.get("reg", None)
        wh = results.get("wh", None)
        batch, channel, height, width = fmap.shape
        K = min(K, height * width)
        if nms:
            fmap = self.pseudo_nms(fmap, output_stride=output_stride)

        scores, index, clses, ys, xs = self.topk_score_in_wh_range(fmap, wh, output_stride=output_stride, K=K)

        if reg is not None:
            reg = gather_feature(reg, index, use_transform=True)
            reg = reg.reshape(batch, K, 2)
            xs_out = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys_out = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs_out = xs.view(batch, K, 1) + 0.5
            ys_out = ys.view(batch, K, 1) + 0.5
        wh = gather_feature(wh, index, use_transform=True)

        # map centers, wh to image size
        xs_out = xs_out * output_stride
        ys_out = ys_out * output_stride
        # in early stage of training, it can be large causing Inf
        wh[:, :, 0].clamp_(max=width)
        wh[:, :, 1].clamp_(max=height)
        wh *= output_stride

        if cat_spec_wh:
            wh = wh.view(batch, K, channel, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            wh = wh.reshape(batch, K, 2)

        clses = clses.reshape(batch, K)
        scores = scores.reshape(batch, K)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        # xyxy
        bboxes = torch.cat([xs_out - half_w, ys_out - half_h,
                            xs_out + half_w, ys_out + half_h],
                           dim=2)
        # filter out empty boxes by wh, unlikely to have false
        nonempty = (wh > 0).all(dim=-1)
        # list of detections per image
        detections = []
        # remove results based on threshold
        # keep = scores > threshold
        for i in range(batch):
            detection_per_image = {}
            bboxes_per_image = bboxes[i, nonempty[i]]
            scores_per_image = scores[i, nonempty[i]]
            clses_per_image = clses[i, nonempty[i]]
            centers_per_image = torch.cat((xs_out[i, nonempty[i]],
                                           ys_out[i, nonempty[i]]), dim=-1)
            keep = scores_per_image > threshold
            bboxes_per_image = bboxes_per_image[keep]
            scores_per_image = scores_per_image[keep]
            clses_per_image = clses_per_image[keep]
            centers_per_image = centers_per_image[keep]

            detection_per_image["pred_boxes"] = bboxes_per_image
            detection_per_image["scores"] = scores_per_image
            detection_per_image["pred_classes"] = clses_per_image
            # centers in image!
            detection_per_image["pred_centers"] = centers_per_image
            detections.append(detection_per_image)
        return detections

    def decode_detections(self, results, output_stride=4, nms=False,
                          cat_spec_wh=False, K=100, threshold=0.0):
        # scores = self.get_scores(results)
        fmap = results.get("cls", None)
        reg = results.get("reg", None)
        wh = results.get("wh", None)

        batch, channel, height, width = fmap.shape
        K = min(K, height * width)
        if nms:
            fmap = self.pseudo_nms(fmap, output_stride=output_stride)

        scores, index, clses, ys, xs = self.topk_score(fmap, K=K)

        if reg is not None:
            reg = gather_feature(reg, index, use_transform=True)
            reg = reg.reshape(batch, K, 2)
            xs_out = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys_out = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs_out = xs.view(batch, K, 1) + 0.5
            ys_out = ys.view(batch, K, 1) + 0.5
        wh = gather_feature(wh, index, use_transform=True)


        # map centers, wh to image size
        xs_out = xs_out * output_stride
        ys_out = ys_out * output_stride
        # in early stage of training, it can be large causing Inf
        wh[:, :, 0].clamp_(max=width)
        wh[:, :, 1].clamp_(max=height)
        wh *= output_stride

        if cat_spec_wh:
            wh = wh.view(batch, K, channel, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            wh = wh.reshape(batch, K, 2)

        clses = clses.reshape(batch, K)
        scores = scores.reshape(batch, K)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        # xyxy
        bboxes = torch.cat([xs_out - half_w, ys_out - half_h,
                            xs_out + half_w, ys_out + half_h],
                           dim=2)
        # filter out empty boxes by wh, unlikely to have false
        nonempty = (wh > 0).all(dim=-1)
        # list of detections per image
        detections = []
        # remove results based on threshold
        # keep = scores > threshold
        for i in range(batch):
            detection_per_image = {}
            bboxes_per_image = bboxes[i, nonempty[i]]
            scores_per_image = scores[i, nonempty[i]]
            clses_per_image = clses[i, nonempty[i]]
            centers_per_image = torch.cat((xs_out[i, nonempty[i]],
                                           ys_out[i, nonempty[i]]), dim=-1)
            keep = scores_per_image > threshold
            bboxes_per_image = bboxes_per_image[keep]
            scores_per_image = scores_per_image[keep]
            clses_per_image = clses_per_image[keep]
            centers_per_image = centers_per_image[keep]

            # nms if necessary
            # returned the index of the kept boxes sorted by scores
            # keep = batched_nms(bboxes_per_image, scores_per_image, clses_per_image, 0.5)
            # if K >= 0:
            #     keep = keep[:K]
            # bboxes_per_image = bboxes_per_image[keep]
            # scores_per_image =scores_per_image[keep]
            # clses_per_image = clses_per_image[keep]
            # centers_per_image = centers_per_image[keep]

            detection_per_image["pred_boxes"] = bboxes_per_image
            detection_per_image["scores"] = scores_per_image
            detection_per_image["pred_classes"] = clses_per_image
            # centers in image!
            detection_per_image["pred_centers"] = centers_per_image
            detections.append(detection_per_image)
        return detections



    def decode_multiscale_relations(self, results, detections, image_sizes, rel_top_k=100):
        rel_scores_all_scale, rel_inds_all_scale = [], []
        for i, (out_stride, size_range) in enumerate(zip(self.output_strides, self.SIZE_RANGE)):
            rel_scores, rel_inds = \
                self.decode_relations_per_scale(results[i], detections, image_sizes,
                                                output_stride=out_stride,
                                                size_range=size_range)
            rel_scores_all_scale.append(rel_scores)
            rel_inds_all_scale.append(rel_inds)
        # perform top k again
        for i in range(len(detections)):
            rel_inds_per_image = torch.cat([rel_inds[i] for rel_inds in rel_inds_all_scale], dim=0)
            rel_scores_per_image = torch.cat([rel_scores[i] for rel_scores in rel_scores_all_scale], dim=0)
            # when we have eps > 0, there may have duplicate relations
            # self._logger.info(np.unique(rel_inds_per_image.detach().cpu().numpy(), axis=1).shape[0])
            rel_top_k = min(rel_top_k, rel_scores_per_image.size(0))
            if rel_top_k > 0:  # in case we have 0 relation
                _, topk_rel_inds = torch.topk(rel_scores_per_image.max(dim=1).values, rel_top_k, dim=0)
                rel_scores_per_image = rel_scores_per_image[topk_rel_inds]
                rel_inds_per_image = rel_inds_per_image[topk_rel_inds]
            detections[i]["rel_scores"] = rel_scores_per_image
            detections[i]["rel_inds"] = rel_inds_per_image
            # add raw raf
            detections[i]["rafs"] = [r['raf'][i] for r in results]

    def decode_relations_per_scale(self, results, detections, image_sizes, output_stride=None, size_range=None):
        """
        results: a dict() contains at least "raf", batched raf
        detection: a list of dict() contains at least "pred_centers"
        """
        batched_rafs = results.get("raf", None)
        # batched_rafs.clamp_(min=-1.0, max=1.0)
        batched_rafs = F.hardtanh(batched_rafs) # torch.tanh(batched_rafs)
        # reshape to (B, P, 2, h, w)
        batched_rafs = batched_rafs.view(batched_rafs.size(0), -1, 2, batched_rafs.size(-2), batched_rafs.size(-1))
        batched_rel_scores, batched_rel_inds = [], []
        for i, detection_per_image in enumerate(detections):
            rafs = batched_rafs[i]
            # get back to feature level
            centers = detection_per_image["pred_centers"] # // output_stride
            pred_classes = detection_per_image["pred_classes"]
            scores = detection_per_image["scores"]
            # based on norm
            # norm_range = torch.tensor((rafs.size(3), rafs.size(2)), dtype=torch.float).norm() \
            #              * torch.tensor(self.SIZE_RANGE[output_stride])

            if self.use_fix_size:
                size_tensor = torch.tensor((self.max_size, self.max_size), dtype=torch.float)
            else:
                size_tensor = torch.tensor(image_sizes[i][::-1], dtype=torch.float)
            size_range = size_tensor.unsqueeze(1) \
                         @ torch.tensor(size_range).unsqueeze(0)
            size_range = size_range.to(self.device)
            # now feed in image level centers
            # (K, 50), (K)
            rel_scores, rel_inds = self.path_integral(rafs, centers,
                                                      scores,
                                                      pred_classes,
                                                      size_range,
                                                      output_stride=output_stride)
            batched_rel_scores.append(rel_scores)
            batched_rel_inds.append(rel_inds)
        return batched_rel_scores, batched_rel_inds

    def path_integral(self,
                      rafs: torch.Tensor,
                      centers: torch.Tensor,
                      scores: torch.Tensor,
                      pred_classes: Optional[torch.Tensor],
                      wh_range: Optional[torch.Tensor],
                      K: int = 100,
                      integral_length: int = 256,
                      integral_width: int = 1,
                      output_stride: int = 4,
                      threshold: float = 0.5):
        """
        Perform line integral over raf based on the detected centers.
        Note that the raf and center should be in the same feature scale.
        rafs: (P, 2, h, w)
        centers: (N, 2)

        OpenPose source codes:
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/net/bodyPartConnectorBase.cpp
        """
        # keep only top objects
        # mask = scores > 0.8
        # centers = centers[mask]
        # scores = scores[mask]
        # centers = centers[:50]
        # scores = scores[:50]
        # some centers will collapse
        rel_inds = torch.combinations(torch.arange(centers.size(0), device=self.device))
        P, _, h, w = rafs.size()

        if wh_range is not None:
            # keep more relations
            # K = 1000
            # get pairs
            centers_pairs = centers[rel_inds]  # (4950, 2, 2), last dim x, y
            # we add a small eps for avoiding zero vectors
            rel_unit_vecs = centers_pairs[:, 1, :] - centers_pairs[:, 0, :] + 1e-12
            rel_norms = rel_unit_vecs.norm(dim=-1)
            # rel_unit_vecs_abs = torch.abs(rel_unit_vecs)
            # valid = torch.logical_and(rel_unit_vecs_abs > wh_range[..., 0],
            #                           rel_unit_vecs_abs <= wh_range[..., 1]).any(dim=-1)
            norm_range = wh_range.norm(dim=0)
            valid = torch.logical_and(rel_norms > norm_range[0],
                                      rel_norms <= norm_range[1])
            # the kept relations index
            rel_inds = rel_inds[valid]
            if not rel_inds.numel():
                return torch.empty((0, P), device=self.device), \
                       torch.empty((0, 2), dtype=torch.long, device=self.device)
        # convert to feature level
        centers = centers // output_stride
        centers_pairs = centers[rel_inds]
        rel_unit_vecs = centers_pairs[:, 1, :] - centers_pairs[:, 0, :] + 1e-12
        rel_norms = rel_unit_vecs.norm(dim=-1)
        rel_unit_vecs = F.normalize(rel_unit_vecs, dim=1)
        N = centers_pairs.size(0)


        integral_length = min(int(rel_norms.max().ceil()), integral_length)
        # integral_length = 3
        # rel_unit_vecs = rel_unit_vecs / rel_norms[..., None]
        ortho_rel_unit_vecs = rel_unit_vecs[:, [1, 0]] * torch.tensor([1, -1], device=self.device)
        # rel_unit_vecs.clamp_(min=-1.0, max=1.0)
        centers_pairs_np = centers_pairs.detach().cpu().numpy()
        # pytorch does not have such linspace capability
        integral_space = np.linspace(centers_pairs_np[:, 0, :],
                                     centers_pairs_np[:, 1, :], num=integral_length)
        # if we have integral width
        # integral_width = int(math.log2(self.max_stride / output_stride) // 2 + 1) * 2 - 1
        integral_space = np.broadcast_to(integral_space, (integral_width,) + integral_space.shape).copy()
        ortho_rel_unit_vecs = ortho_rel_unit_vecs.unsqueeze(0).expand(integral_width, -1, -1).clone()
        shifts = torch.arange(-(integral_width // 2), integral_width // 2 + 1, device=self.device)[:, None, None]
        integral_space = integral_space + (ortho_rel_unit_vecs * shifts).unsqueeze(1).detach().cpu().numpy()
        # round to closest pixel
        integral_space = np.rint(integral_space).astype(np.int32)
        integral_space = np.clip(integral_space, [0, 0], [w - 1, h - 1])

        # normalize each channel of RAFs by the maximum norm
        # max_norms = rafs.norm(dim=1).view(P, -1).max(dim=1).values.clamp(min=1e-12)
        # rafs = rafs / max_norms[:, None, None, None]
        # fancy indexing, bascially we are selecting all batch, all predicates,
        # but spatial differently, shape is (P, 2, INTEGRAL_WIDTH, INTEGRAL_LENGTH, N)
        rafs = rafs[:, :, integral_space[:, :, :, 1], integral_space[:, :, :, 0]]
        # reshape to (N, P, INTEGRAL_WIDTH, INTEGRAL_LENGTH, 2)
        rafs = rafs.permute(4, 0, 2, 3, 1)
        # (N, P, INTEGRAL_WIDTH, 2, 1)
        rel_unit_vecs = rel_unit_vecs.unsqueeze(1).unsqueeze(1).expand(-1, P, integral_width, -1).clone().unsqueeze(-1)
        # shape (N, P, INTEGRAL_WIDTH, INTEGRAL_LENGTH, 1) =
        # (N, P, INTEGRAL_WIDTH, INTEGRAL_LENGTH, 2) x (N, P, INTEGRAL_WIDTH, 2, 1)
        rel_scores = torch.matmul(rafs, rel_unit_vecs).view(N, P, -1)

        # 80% criterion
        # keep = torch.logical_or(
        #     (rel_scores >= 0.8).sum(dim=-1).max(dim=-1).values >= integral_length * 0.8,
        #     (rel_scores <= -0.8).sum(dim=-1).max(dim=-1).values >= integral_length * 0.8)
        # rel_scores = rel_scores[keep]
        # rel_inds = rel_inds[keep]

        # by averaging
        # rel_scores = rel_scores.sum(dim=[-1])
        rel_scores = rel_scores.mean(dim=[-1])
        # rel_scores.clamp_(min=-1.0, max=1.0)

        # weighted by vector length
        # rel_scores = rel_scores * (1 / rel_norms[:, None])

        # filter/weight by std?
        # rel_scores = rel_scores.mean(dim=[-1]) / (rel_scores.std(dim=[-1]) + 1e-12)

        # filter low response rels
        # response_scores = torch.abs(rel_scores).sum(dim=1)
        # _, top_response_inds = torch.topk(response_scores, min(10 * K, response_scores.size(0)))
        # rel_scores, rel_inds = rel_scores[top_response_inds], rel_inds[top_response_inds]

        # handle negative scores, so for a pair of objects, over P predicates,
        # some could be positive (s->o) and some could be negative (o->s)
        pos_scores = torch.where(rel_scores >= 0.0, rel_scores, torch.tensor(0.0, device=self.device))
        neg_scores = torch.where(rel_scores < -0.0, -rel_scores, torch.tensor(0.0, device=self.device))
        # or for a pair of objects, if we only keep the max integral no matter which one is subj/obj
        # mask = pos_scores.max(dim=1)[0] > neg_scores.max(dim=1)[0]
        # rel_scores = torch.where(mask[:, None], pos_scores, neg_scores)
        # rel_inds = torch.where(mask[:, None], rel_inds, rel_inds[:, [1, 0]])
        rel_scores = torch.cat((pos_scores, neg_scores))
        rel_inds = torch.cat((rel_inds, rel_inds[:, [1, 0]]))
        # only keep by threshold
        # keep = rel_scores.max(dim=1).values > 0.8
        # rel_scores = rel_scores[keep]
        # rel_inds = rel_inds[keep]

        # we can weight the relation score by the object detection scores
        obj_scores = scores[rel_inds]
        obj_scores = obj_scores.prod(1, keepdim=True)
        rel_scores = rel_scores * obj_scores
        if self.freq_bias is not None:
            cls_pairs = pred_classes[rel_inds]
            self.freq_bias = self.freq_bias.to(self.device)
            freq = self.freq_bias[cls_pairs[:,0], cls_pairs[:,1], :]
            rel_scores = freq * rel_scores # torch.where(freq > 0, rel_scores * torch.sqrt(freq), rel_scores)
        # we do top k by max over P predicates for each relation
        # so there are (# rels, 1) scores where one score for one relation
        K = min(K, rel_scores.size(0))
        if K > 0:  # in case we have 0 relation
            _, topk_rel_inds = torch.topk(rel_scores.max(dim=1).values, K, dim=0)
            # _, topk_rel_inds = torch.topk(rel_scores.sum(dim=1), K, dim=0)
            rel_scores, rel_inds = rel_scores[topk_rel_inds], rel_inds[topk_rel_inds]
        return rel_scores, rel_inds


    def decode_relations(self, results, detections, K=100, output_stride=4):
        """
        Process raf output of the network per image, since bboxes are
        filtered first, then there will be variable number of bboxes for
        each image.
        Number of predicate classes P.
        Number of total predictions allowed per image N.
        rafs: Relation Affinity Fields of shape (P, 2, h, w)
        centers_x: The predicted object centers x-coordinate, shape (N,)
        centers_y: y-coordinate.
        scores: the scores of center predictions, (N,)

        We need to prepare these for relation evaluations:
        rel_pair_idxs (#pred_rels, 2)
        pred_rel_scores (#pred_rels, num_pred_class), num_pred_class = 50 for VG
        """
        batched_rafs = results.get("raf", None)
        # batched_rafs.clamp_(min=-1, max=1)
        if len(batched_rafs.size()) == 4:
            b, c, h, w = batched_rafs.size()
            # reshape to (B, P, 2, h, w)
            batched_rafs = batched_rafs.view(b, -1, 2, h, w)
        else:
            c = batched_rafs.size(1) * 2
        for i, detection_per_image in enumerate(detections):
            rafs = batched_rafs[i]
            # get back to feature level, float tensor
            centers = detection_per_image["pred_centers"]
            scores = detection_per_image["scores"]
            if scores.numel():
                rel_scores, rel_inds = self.path_integral(rafs, centers, scores,
                                                          None, None, output_stride=output_stride)
            else:
                rel_scores = torch.empty((0, c // 2), device=scores.device)
                rel_inds = torch.empty((0, 2), device=scores.device)
            detection_per_image["rel_scores"] = rel_scores
            detection_per_image["rel_inds"] = rel_inds


    def decode_point_relations(self, results, detections, K=100, output_stride=4):
        """
        Process raf output of the network per image, since bboxes are
        filtered first, then there will be variable number of bboxes for
        each image.
        Number of predicate classes P.
        Number of total predictions allowed per image N.
        rafs: Relation Affinity Fields of shape (P, 2, h, w)
        centers_x: The predicted object centers x-coordinate, shape (N,)
        centers_y: y-coordinate.
        scores: the scores of center predictions, (N,)

        We need to prepare these for relation evaluations:
        rel_pair_idxs (#pred_rels, 2)
        pred_rel_scores (#pred_rels, num_pred_class), num_pred_class = 50 for VG
        """
        batched_rafs = results.get("raf", None)
        # batched_rafs.clamp_(min=-1, max=1)
        if len(batched_rafs.size()) == 4:
            b, c, h, w = batched_rafs.size()
            # reshape to (B, P, 2, h, w)
            batched_rafs = batched_rafs.view(b, -1, 2, h, w)
        else:
            c = batched_rafs.size(1) * 2
        for i, detection_per_image in enumerate(detections):
            rafs = batched_rafs[i]
            # get back to feature level, float tensor
            centers = detection_per_image["pred_centers"]
            scores = detection_per_image["scores"]
            if scores.numel():
                rel_scores, rel_inds = self.rel_gather(rafs, centers, scores, output_stride=output_stride)
            else:
                rel_scores = torch.empty((0, c // 2), device=scores.device)
                rel_inds = torch.empty((0, 2), device=scores.device)
            detection_per_image["rel_scores"] = rel_scores
            detection_per_image["rel_inds"] = rel_inds

    def rel_gather(self,
                      rafs: torch.Tensor,
                      centers: torch.Tensor,
                      scores: torch.Tensor,
                      output_stride: int = 4,
                      K: int = 100,):
        # keep only top objects
        # centers = centers[:50]
        # scores = scores[:50]
        rel_inds = torch.combinations(torch.arange(centers.size(0), device=self.device))
        centers = centers // output_stride
        centers_pairs = centers[rel_inds].long()
        # rel_unit_vecs = centers_pairs[:, 1, :] - centers_pairs[:, 0, :] + 1e-12
        subj_scores = torch.cat((rafs[:, 0, centers_pairs[:, 0, 1], centers_pairs[:, 0, 0]],
                                 rafs[:, 0, centers_pairs[:, 1, 1], centers_pairs[:, 1, 0]]), dim=1)
        obj_scores = torch.cat((rafs[:, 1, centers_pairs[:, 1, 1], centers_pairs[:, 1, 0]],
                                rafs[:, 1, centers_pairs[:, 0, 1], centers_pairs[:, 0, 0]]), dim=1)
        rel_scores = subj_scores * obj_scores
        rel_scores.transpose_(1, 0)
        # we can weight the relation score by the object detection scores
        rel_inds = torch.cat((rel_inds, rel_inds[:, [1, 0]]))
        obj_scores = scores[rel_inds]
        obj_scores = obj_scores.prod(1, keepdim=True)
        rel_scores = rel_scores * obj_scores
        # we do top k by max over P predicates for each relation
        # so there are (# rels, 1) scores where one score for one relation
        K = min(K, rel_scores.size(0))
        if K > 0:  # in case we have 0 relation
            _, topk_rel_inds = torch.topk(rel_scores.max(dim=1).values, K, dim=0)
            # _, topk_rel_inds = torch.topk(rel_scores.sum(dim=1), K, dim=0)
            rel_scores, rel_inds = rel_scores[topk_rel_inds], rel_inds[topk_rel_inds]
        return rel_scores, rel_inds


    def pseudo_nms(self, fmap, output_stride=4, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pool_size = pool_size if output_stride <= 4 else 1
        # pool_size = 1
        p = max(0, (pool_size - 1) // 2)
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=p)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    def get_scores(self, results, K=100):
        fmaps = []
        for i, (result, stride) in enumerate(zip(results, self.output_strides)):
            fmap = result.get("cls", None)
            batch, channel, height, width = fmap.shape
            fmap = fmap.reshape(batch, -1)
            fmaps.append(fmap)
        # no fmaps[
        #
        #
        fmaps = torch.cat(fmaps, dim=-1)
        scores, inds = torch.topk(fmaps, K)
        return fmaps

    def topk_score_in_wh_range(self, scores, wh, output_stride, K=100):

        batch, channel, height, width = scores.shape
        wh = wh.view(batch, 2, -1)

        range_wh = (torch.tensor((width, height), dtype=torch.float).unsqueeze(
            1) \
                    @ torch.tensor(self.SIZE_RANGE[output_stride]).unsqueeze(0)).to(
            device=self.device)
        # based on either side
        # valid = torch.logical_and(wh >= range_wh[None, :, 0, None],
        #                           wh <= range_wh[None, :, 1, None]).any(dim=1)
        # Based on region area
        gt_area = wh.prod(dim=1)
        range_area = range_wh.prod(dim=0)
        valid = torch.logical_and(gt_area >= range_area[0], gt_area <= range_area[1])
        # area_valid = torch.logical_and(gt_area >= range_area[0], gt_area <= range_area[1])
        # valid = torch.logical_or(valid, area_valid)


        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1) * (valid[:, None, :].float()), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = index // K
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def topk_score(self, scores, K=100):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = index // K
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def detector_postprocess(results, relation_on=False):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (dict): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.

    input_width = results.pop("input_width", None)
    input_height = results.pop("input_height", None)
    output_width = results.pop("output_width", None)
    output_height = results.pop("output_height", None)
    # convert to Boxes
    if "pred_boxes" in results:
        results["pred_boxes"] = Boxes(results["pred_boxes"])
    # boxes_per_image = results.get("bboxes", None)
    # scores_per_image = results.get("scores", None)
    # classes_per_image = results.get("clses", None)
    rel_scores_per_image = results.pop("rel_scores", None)
    rel_inds_per_image = results.pop("rel_inds", None)
    pred_rafs = results.pop("rafs", None)

    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / input_width,
        output_height_tmp / input_height,
    )
    # can be Instances or SceneGraph
    instances = Instances((output_height, output_width), **results)

    if instances.has("pred_boxes"):
        output_boxes = instances.pred_boxes
    elif instances.has("proposal_boxes"):
        output_boxes = instances.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(instances.image_size)


    if instances.has("pred_keypoints"):
        instances.pred_keypoints[:, :, 0] *= scale_x
        instances.pred_keypoints[:, :, 1] *= scale_y
    scene_graph = None
    if relation_on:
        scene_graph = SceneGraph((output_height, output_width), **results)
        scene_graph.rel_scores = rel_scores_per_image
        scene_graph.rel_inds = rel_inds_per_image

        # map to triplets
        pred_classes = results.get("pred_classes")
        pred_scores = results.get("scores")
        pred_rel_scores, pred_predicates = rel_scores_per_image.max(dim=1, keepdim=True)
        # (N, 3 + 3 + 4 * 2 + 2 = 16)
        # predicted graph of subj class, obj class, predicate class,
        #                    subj scores, object scores, predicate scores,
        #                    subj box, obj box, subj id, obj id
        pred_graph = torch.cat((pred_classes[rel_inds_per_image],
                                pred_predicates,
                                pred_scores[rel_inds_per_image],
                                pred_rel_scores,
                                output_boxes[rel_inds_per_image[:, 0]].tensor,
                                output_boxes[rel_inds_per_image[:, 1]].tensor,
                                rel_inds_per_image), dim=1)
        # merge all pred_rafs
        if type(pred_rafs) == list:
            outputs = []
            # determine the stride
            stride = int(max(input_width / pred_rafs[0].size(3), input_height / pred_rafs[0].size(2)))
            stride = int(math.log2(stride))
            for i, pred_raf in enumerate(pred_rafs):
                outputs.append(F.interpolate(pred_raf,
                                             scale_factor=2 ** (stride + i),
                                             mode="nearest",
                                             align_corners=None))

            pred_rafs = sum(outputs)
            # clip paddings
            pred_rafs = pred_rafs[:, :, :input_height, :input_width]
            # then scale to the target size
            pred_rafs = F.interpolate(pred_rafs,
                                             size=(output_height_tmp, output_width_tmp),
                                             mode="nearest",
                                             align_corners=None)

        scene_graph.pred_graph = pred_graph
        scene_graph.pred_rafs = pred_rafs

    return scene_graph, instances