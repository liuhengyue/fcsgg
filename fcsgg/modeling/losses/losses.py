"""
Core implementations of the losses used in FCSGG:
    gaussian_focal_loss - center heatmaps
    RAFLoss - a class wrapper to compute the loss of Relation Affinity Fields
    CB_loss_weights - Class-Balanced Loss (https://arxiv.org/abs/1901.05555)
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["mmdetection", "https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.data import MetadataCatalog, DatasetCatalog

__all__ = ["RAFLoss", "RelationLoss"]

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    From mmdetection:
    hhttps://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
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

def CB_loss_weights(samples_per_cls, beta):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    no_of_classes = len(samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    return weights


class RAFLoss(nn.Module):
    def __init__(self,
                 cfg,
                 cos_similar=False,
                 beta=0.999,
                 reduction='mean',
                 loss_weight=1.0):
        super(RAFLoss, self).__init__()
        self.raf_type = cfg.INPUT.RAF_TYPE
        self.cos_similar = cos_similar
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_type, self.reg_area, self.class_balance, self.neg_loss_weight = cfg.MODEL.HEADS.RAF.LOSS_TYPE
        if self.class_balance == 'cb':
            try:
                predicate_stats = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).predicate_stats
                self.cb_weight = torch.as_tensor(CB_loss_weights(predicate_stats, beta))
                # will broadcast to (b, 50, 2, h, w)
                self.cb_weight = self.cb_weight[None, :, None, None, None]
                self.class_balance = True
            except AttributeError:  # predicate stats are not available for this dataset
                self.class_balance = False
        else:
            self.class_balance = False
        if self.reg_type == "l1":
            self.loss_func = F.l1_loss
        elif self.reg_type == "l2":
            self.loss_func = F.mse_loss
        elif self.reg_type == "smooth_l1":
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImplemented()

    def _cosine_similarity_loss(self, preds, targets, weights, num=None):
        weights = weights[:, :, 0, ...]
        # num = weights.gt(0).sum().clamp(min=1)
        # batched_num_valid = weights.gt(0).sum(dim=[1, 2, 3]).clamp(min=1)  # shape (b, )
        cosine_sim = F.cosine_similarity(preds, targets, dim=2)
        # no gt position will be zero
        loss = (1 - cosine_sim) * weights
        # loss = (loss.sum(dim=[1, 2, 3]) / batched_num_valid).sum()
        # valid = (weights > 0).any(dim=2)
        loss = loss.sum() / num
        # cosine_sim = F.cosine_similarity(preds, targets, dim=2)[valid]
        # loss = F.l1_loss(cosine_sim, torch.ones_like(cosine_sim), reduction='none')
        return loss

    def _reg_loss(self, preds, targets, weights, num=None, weighted=True):
        # in case we have no relation, the loss will still be zero since gt_raf_weights are all zero
        # num = weights.gt(0).sum().clamp(min=1)
        num = weights.eq(1).sum().clamp(min=1)
        if weighted:
            loss = self.loss_func(preds, targets, reduction='none') * weights
        else:
            loss = self.loss_func(preds, targets, reduction='none') * (weights > 0).float()
        if self.cos_similar:
            angle = torch.pow(-F.cosine_similarity(preds, targets, dim=2) + 2.0, 2)
            loss = loss * angle[:, :, None, ...]
        if self.class_balance:
            loss = loss * self.cb_weight.to(preds.device)
        loss = loss.sum() / num
        return loss

    def forward(self, preds, targets):
        gt_rafs = torch.stack([x.gt_relations for x in targets], dim=0)
        # num_rels = max(sum([x.gt_num_relations for x in targets]), 1)
        # preds may be of (B, P*2, h, w)
        preds = preds.view_as(gt_rafs)
        gt_raf_weights = torch.stack([x.gt_relations_weights for x in targets], dim=0)
        if self.raf_type == "vector":
            loss = self._reg_loss(preds, gt_rafs, gt_raf_weights, num=None)
            loss = loss * self.loss_weight

            if self.reg_area == "all":
                neg_mask = gt_raf_weights == 0
                # # uncomment L147-150 if we want to randomly select some negative samples for training
                # rand_mask = torch.empty_like(gt_raf_weights, dtype=torch.bool).random_(2)
                # # positive location stays False
                # neg_mask = torch.logical_and(neg_mask, rand_mask)
                # regularization on other locations without GT
                loss += F.l1_loss(preds[neg_mask],
                                  gt_rafs[neg_mask],
                                  reduction='mean') * self.neg_loss_weight
            elif self.reg_area == "neg":
                # spatial weights (class-agnostic) on gt paths
                spatial_mask = gt_raf_weights.max(dim=1, keepdim=True).values != gt_raf_weights
                loss += (F.l1_loss(preds, gt_rafs, reduction='none') * spatial_mask ).mean() * self.neg_loss_weight

        elif self.raf_type == "point":
            loss = gaussian_focal_loss(preds, gt_rafs)
        else:
            raise NotImplementedError()
        return loss


class AdjustSmoothL1Loss(nn.Module):

    def __init__(self, num_features, momentum=0.1, beta=1. /9):
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs - target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))


        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

class RelationLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=0.1,
                 ce=False):
        super(RelationLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ce =ce

    def _line_integral(self,
                       rafs: torch.Tensor,
                       batch_inds: torch.Tensor,
                       subj_centers: torch.Tensor,
                       obj_centers: torch.Tensor,
                       predicates: torch.Tensor):
        """
        Note the input centers here are in the order of x, y
        """
        # no relation
        if predicates.numel() == 0:
            return 0.0 * rafs.sum()
        rafs = rafs.clamp(min=-1, max=1)
        b, _, h, w = rafs.size()
        rafs = rafs.view(b, -1, 2, h, w)
        # first inverse it
        rel_unit_vecs = (obj_centers - subj_centers).float()
        rel_norms = rel_unit_vecs.norm(dim=-1)
        # filter overlapped subject object
        valid = torch.nonzero(rel_norms, as_tuple=True)[0]
        if valid.numel() == 0:
            return torch.empty((0), device=valid.device), valid
        batch_inds = batch_inds[valid]
        predicates = predicates[valid]
        rel_unit_vecs = rel_unit_vecs[valid]
        rel_norms = rel_norms[valid]
        obj_centers = obj_centers[valid]
        subj_centers = subj_centers[valid]
        # (# rels, 2)
        rel_unit_vecs = rel_unit_vecs / rel_norms[..., None]

        obj_centers_np = obj_centers.detach().cpu().numpy()
        subj_centers_np = subj_centers.detach().cpu().numpy()
        rel_scores = []
        for i in range(valid.size(0)):
            integral_space = np.linspace(obj_centers_np[i],
                                         subj_centers_np[i],
                                         num=int(rel_norms[i].ceil()))
            integral_space = np.rint(integral_space).astype(np.int32)
            integral_space = np.unique(integral_space, axis=0)
            if self.ce:
                # (P, 2, N)
                rafs_per_rel = rafs[batch_inds[i],
                               :, :,
                               integral_space[:, 1],
                               integral_space[:, 0]]
                # (P)
                rel_score = (rafs_per_rel.permute(0, 2, 1) * rel_unit_vecs[None, None, i]).sum(dim=-1).mean(dim=1)

            else:
                rafs_per_rel = rafs[batch_inds[i],
                               predicates[i], :,
                               integral_space[:, 1],
                               integral_space[:, 0]]
                rel_score = torch.matmul(rel_unit_vecs[i], rafs_per_rel).mean()
            rel_scores.append(rel_score)
        rel_scores = torch.stack(rel_scores)
        return rel_scores, valid


    def forward(self, rafs, heatmaps, targets):
        # generate 4D index for each gt in the order of B, C, h, w
        # list of (#rel, 3)
        center_inds = [torch.stack((x.gt_classes,
                              x.gt_centers_int[:, 1],
                              x.gt_centers_int[:, 0]), dim=1)
                 for i, x in enumerate(targets)]
        # (#GT, 4)
        # Get relations in the order of B, subj, obj, predicate
        gt_rels = [x.get_extra("gt_relations") for x in targets]
        # a index tensor of shape (N, 8)
        rels = [torch.cat((torch.ones(x.size(0),
                            dtype=torch.long, device=x.device).unsqueeze(1) * i, # batch idx
                            center_inds[i][x[:, 0]], # subject cls, y, x
                            center_inds[i][x[:, 1]], # object cls, y, x
                            x[:, 2, None]), dim=1) # predicate class
                            for i, x in enumerate(gt_rels)]
        rels = torch.cat(rels, dim=0)
        if rels.numel():
            # get the subject and object class scores
            subj_cls_scores = heatmaps[rels[:, 0], rels[:, 1], rels[:, 2], rels[:, 3]]
            obj_cls_scores = heatmaps[rels[:, 0], rels[:, 4], rels[:, 5], rels[:, 6]]
            # get the integral
            integral_scores, valid = self._line_integral(rafs,
                                                  rels[:, 0],
                                                  rels[:, 2:4][:, [1, 0]],
                                                  rels[:, 5:7][:, [1, 0]],
                                                  rels[:, 7])
            if valid.numel():
                subj_cls_scores = subj_cls_scores[valid, None]
                obj_cls_scores = obj_cls_scores[valid, None]
                # integral_scores for GT should always be positive, penalize neg values
                rel_scores = subj_cls_scores * obj_cls_scores * integral_scores.clamp(min=0, max=1)
                if self.ce:
                    loss = F.cross_entropy(rel_scores, rels[:, 7][valid]) * self.loss_weight
                else:
                    loss = F.binary_cross_entropy(rel_scores, torch.ones_like(rel_scores),
                                              reduction=self.reduction) * self.loss_weight
            else:
                loss = 0.0 * rafs.sum()
        else:
            loss = 0.0 * rafs.sum()
        return loss