"""
Scene graph evaluation codes that support multiple metrics.

Modified to be compatible with Detectron2 based on
https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Kaihua Tang"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import logging
import os
import torch
import numpy as np
import json
import time
from tqdm import tqdm
from collections import defaultdict
from functools import reduce

from abc import ABC, abstractmethod

__all__ = ["SceneGraphAPI", "SGRecall", "SGNoGraphConstraintRecall", "SGZeroShotRecall", "SGNGZeroShotRecall",
           "SGPairAccuracy", "SGMeanRecall", "SGNGMeanRecall", "SGAccumulateRecall"]


class SceneGraphAPI:
    def __init__(self, cfg, annotation_file=None, logger=None):
        """
        Constructor of Visual Genome helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.num_rel_category = cfg.MODEL.HEADS.NUM_PREDICATES
        self.multiple_preds = cfg.RELATION.MULTIPLE_PREDS
        self.iou_thres = cfg.RELATION.IOU_THRESHOLD
        self.use_gt_box = cfg.RELATION.USE_GT_BOX
        self.use_gt_object_label = cfg.RELATION.USE_GT_OBJECT_LABEL
        self.logger = logger
        # if self.use_gt_box:
        #     if self.use_gt_object_label:
        #         self.mode = 'predcls'
        #     else:
        #         self.mode = 'sgcls'
        # else:
        #     self.mode = 'sgdet'
        # load dataset
        self.dataset,self.rel_cats,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns = dict()
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        cats, imgs, rel_cats, imgToAnns = {}, {}, {}, {}
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['image_id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'predicate_categories' in self.dataset:
            for cat in self.dataset['predicate_categories']:
                rel_cats[cat['id']] = cat

        print('index created!')

        # create class members
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.cats = cats
        self.rel_cats = rel_cats


    def get_evaluators(self):
        # feed other info
        global_container = {}
        global_container['zeroshot_triplet'] = torch.load("fcsgg/evaluation/zeroshot_triplet.pytorch",
                                                          map_location=torch.device("cpu")).long().numpy()
        global_container['mode'] = self.mode
        global_container['multiple_preds'] = self.multiple_preds
        global_container['num_rel_category'] = self.num_rel_category
        global_container['iou_thres'] = self.iou_thres
        rel_name_list = [val["name"] for val in self.rel_cats.values()]
        result_dict = {}
        evaluators = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict, self.mode)
        evaluators['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict, self.mode)
        evaluators['eval_nog_recall'] = eval_nog_recall
        if global_container['zeroshot_triplet'] is not None:
            # test on different distribution
            eval_zeroshot_recall = SGZeroShotRecall(result_dict, self.mode)
            evaluators['eval_zeroshot_recall'] = eval_zeroshot_recall

            # test on no graph constraint zero-shot recall
            eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict, self.mode)
            evaluators['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

            # test on mean zero-shot recall
            eval_zeroshot_mean_recall = SGMeanZeroShotRecall(result_dict, self.mode,
                                                           num_rel=self.num_rel_category,
                                                           ind_to_predicates=rel_name_list,
                                                           print_detail=True)
            evaluators['eval_zeroshot_mean_recall'] = eval_zeroshot_mean_recall

            # test on ng mean zero-shot recall
            eval_ng_zeroshot_mean_recall = SGNGMeanZeroShotRecall(result_dict, self.mode,
                                                             num_rel=self.num_rel_category,
                                                             ind_to_predicates=rel_name_list,
                                                             print_detail=True)
            evaluators['eval_ng_zeroshot_mean_recall'] = eval_ng_zeroshot_mean_recall

        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict, self.mode)
        evaluators['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, self.mode,
                                        num_rel=self.num_rel_category,
                                        ind_to_predicates=rel_name_list,
                                        print_detail=True)
        evaluators['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, self.mode,
                                             num_rel=self.num_rel_category,
                                             ind_to_predicates=rel_name_list,
                                             print_detail=True)
        evaluators['eval_ng_mean_recall'] = eval_ng_mean_recall


        global_container['result_dict'] = result_dict
        global_container['evaluators'] = evaluators

        return global_container


    def evaluate(self, predictions, task):
        assert task in ["sgdet", "sgcls", "predcls"], f"{task} task is not supported."
        assert (task in ["sgcls"]) == self.use_gt_box, f"{task} task has to set RELATION.USE_GT_BOX to {self.use_gt_box}, " \
                                                     f"since the gt has to be changed before the evaluation."
        assert (task == "predcls") == self.use_gt_object_label, f"{task} task has to set RELATION.USE_GT_BOX to {self.use_gt_object_label}, " \
                                                     f"since the gt has to be changed before the evaluation."
        self.mode = task
        # reset results dict
        self.results = self.get_evaluators()
        gt_annotations = [self.imgToAnns[pred["image_id"]] for pred in predictions]
        for gt_annotation, pred in zip(gt_annotations, predictions):
            self.evaluate_relation_of_one_image(gt_annotation, pred)
        evaluators = self.results["evaluators"]
        # calculate mean recall
        evaluators["eval_mean_recall"].calculate_mean_recall()
        evaluators["eval_ng_mean_recall"].calculate_mean_recall()
        evaluators["eval_zeroshot_mean_recall"].calculate_mean_recall()
        evaluators["eval_ng_zeroshot_mean_recall"].calculate_mean_recall()

        # print result
        result_str = '=' * 100 + '\n'
        result_str += evaluators["eval_recall"].generate_print_string()
        result_str += evaluators["eval_nog_recall"].generate_print_string()
        result_str += evaluators["eval_zeroshot_recall"].generate_print_string()
        result_str += evaluators["eval_ng_zeroshot_recall"].generate_print_string()
        result_str += evaluators["eval_mean_recall"].generate_print_string()
        result_str += evaluators["eval_ng_mean_recall"].generate_print_string()
        result_str += evaluators["eval_zeroshot_mean_recall"].generate_print_string()
        result_str += evaluators["eval_ng_zeroshot_mean_recall"].generate_print_string()

        if self.use_gt_box:
            result_str += evaluators["eval_pair_accuracy"].generate_print_string()
        result_str += '=' * 100 + '\n'
        if self.logger:
            self.logger.info(result_str)
        else:
            print(result_str)

        return {"not implemented yet": 0}

    def evaluate_relation_of_one_image(self, groundtruth, prediction):
        """
        Returns:
            pred_to_gt: Matching from predicate to GT
            pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
            pred_triplet_scores: [cls_0score, relscore, cls1_score]
        """
        # unpack all inputs
        global_container = self.results
        evaluator = self.results["evaluators"]
        mode = global_container['mode']

        assert groundtruth["image_id"] == prediction["image_id"], "Gt and prediction image are mismatched."
        local_container = {}
        local_container['gt_rels'] = np.array(groundtruth.pop('gt_rels'), dtype=np.int32)

        # if there is no gt relations for current image, then skip it
        if len(local_container['gt_rels']) == 0:
            return

        local_container['gt_boxes'] = np.array(groundtruth.pop('gt_boxes'), dtype=np.float32)  # (#gt_objs, 4)
        local_container['gt_classes'] = np.array(groundtruth.pop('gt_classes'), dtype=np.int32)  # (#gt_objs, )

        # about relations
        local_container['pred_rel_inds'] = np.array(prediction.pop('rel_inds'), dtype=np.int32)  # (#pred_rels, 2)
        local_container['rel_scores'] = np.array(prediction.pop('rel_scores'), dtype=np.float32) # (#pred_rels, num_pred_class)

        # about objects

        # for predcls, we set label and score to groundtruth
        if mode == 'predcls':
            # since we do not have 'bg' class nor proposals, we just collect the bounding boxes from gt
            # note these fields are already changed to gt before `decode`.
            local_container['pred_boxes'] = local_container['gt_boxes']  # (#pred_objs, 4)
            local_container['pred_classes'] = local_container['gt_classes']
            local_container['obj_scores'] = np.ones(local_container['pred_classes'].shape[0], dtype=np.float32)
        else:
            local_container['pred_boxes'] = np.array(prediction.pop('bboxes'), dtype=np.float32)  # (#pred_objs, 4)
            local_container['pred_classes'] = np.array(prediction.pop('category_ids'), dtype=np.int32)  # (#pred_objs, )
            local_container['obj_scores'] = np.array(prediction.pop('scores'), dtype=np.float32)  # (#pred_objs, )


        # to calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
        # for sgcls and predcls
        if mode != 'sgdet':
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

        # to calculate the prior label based on statistics
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
        evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
        evaluator['eval_zeroshot_mean_recall'].prepare_zeroshot(global_container, local_container)
        evaluator['eval_ng_zeroshot_mean_recall'].prepare_zeroshot(global_container, local_container)

        if mode == 'sgcls':
            assert local_container['gt_boxes'].shape[0] == local_container['pred_boxes'].shape[0],\
                'Num of GT boxes is not matching with num of pred boxes in SGCLS'


        if local_container['pred_rel_inds'].shape[0] == 0:
            return

        # Traditional Metric with Graph Constraint
        # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
        local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container)
        # No Graph Constraint
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container)
        # GT Pair Accuracy
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container)
        # Mean Recall
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container)
        # No Graph Constraint Mean Recall
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container)
        # Zero shot Recall
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container)
        # No Graph Constraint Zero-Shot Recall
        evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container)
        # Mean zero-shot Recall
        evaluator['eval_zeroshot_mean_recall'].collect_mean_recall_items(global_container, local_container)
        # NG Mean zero-shot Recall
        evaluator['eval_ng_zeroshot_mean_recall'].collect_mean_recall_items(global_container, local_container)
        return


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict, mode, *, num_rel=50, ind_to_predicates=[], print_detail=False):
        super().__init__()
        self.result_dict = result_dict
        self.mode = mode
        self.num_rel = num_rel
        self.rel_name_list = ind_to_predicates
        self.print_detail = print_detail
        self.register_container()

    @abstractmethod
    def register_container(self):
        raise NotImplementedError("You are not supposed to call an abstract class.")

    @abstractmethod
    def generate_print_string(self):
        raise NotImplementedError("You are not supposed to call an abstract class.")


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % self.mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container):
        pred_rel_inds = local_container['pred_rel_inds']
        # (0 - 1, 2 - 3, ... )
        # (9900, 50)
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))
        # (9900, 1)
        pred_scores = rel_scores.max(1)
        # (9900, 9900, 1)
        # (10, 10, 1)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet= self.mode == 'phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[self.mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[self.mode + '_recall'][k].append(rec_i)

        return local_container


"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % self.mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        # obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        # nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores
        nogc_overall_scores = rel_scores
        # (9900, 50)
        # (9900 * 50 ,1)
        # (100, 1)
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1]))
        nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1]]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
            nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet= self.mode == 'phrdet',
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[self.mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[self.mode + '_recall_nogc'][k].append(rec_i)

        return local_container


"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""


class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % self.mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        zeroshot_triplets = zeroshot_triplets - 1

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container, local_container):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[self.mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[self.mode + '_zeroshot_recall'][k].append(zero_rec_i)


"""
No Graph Constraint Mean Recall
"""


class SGNGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_ng_zeroshot_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_ng_zeroshot_recall'].items():
            result_str += 'ng-zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero Shot Recall.' % self.mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        zeroshot_triplets = zeroshot_triplets - 1

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container, local_container):
        pred_to_gt = local_container['nogc_pred_to_gt']

        for k in self.result_dict[self.mode + '_ng_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[self.mode + '_ng_zeroshot_recall'][k].append(zero_rec_i)


class SGMeanZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGMeanZeroShotRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_zeroshot_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[self.mode + '_zeroshot_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)],
                                                           50: [[] for i in range(self.num_rel)],
                                                           100: [[] for i in range(self.num_rel)]}
        self.result_dict[self.mode + '_zeroshot_mean_recall_list'] = {20: [], 50: [], 100: []}

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        zeroshot_triplets = zeroshot_triplets - 1

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_zeroshot_mean_recall'].items():
            result_str += '  zmR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Zero-shot Mean Recall.' % self.mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[self.mode + '_zeroshot_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        if len(self.zeroshot_idx) > 0:
            for k in self.result_dict[self.mode + '_zeroshot_mean_recall_collect']:
                # the following code are copied from Neural-MOTIFS
                match = reduce(np.union1d, pred_to_gt[:k])
                # NOTE: by kaihua, calculate Mean Recall for each category independently
                # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
                recall_hit = [0] * self.num_rel
                recall_count = [0] * self.num_rel
                for idx in range(gt_rels.shape[0]):
                    if idx in self.zeroshot_idx:
                        local_label = gt_rels[idx, 2]
                        recall_count[int(local_label)] += 1
                        recall_count[0] += 1

                for idx in range(len(match)):
                    matched_idx = int(match[idx])
                    # check if matched gt is the zero-shot
                    if matched_idx in self.zeroshot_idx:
                        local_label = gt_rels[matched_idx, 2]
                        recall_hit[int(local_label)] += 1
                        recall_hit[0] += 1

                for n in range(self.num_rel):
                    if recall_count[n] > 0:
                        self.result_dict[self.mode + '_zeroshot_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self):
        for k, v in self.result_dict[self.mode + '_zeroshot_mean_recall'].items():
            sum_recall = 0
            # we do not have 'bg' class
            for idx in range(self.num_rel):
                if len(self.result_dict[self.mode + '_zeroshot_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[self.mode + '_zeroshot_mean_recall_collect'][k][idx])
                self.result_dict[self.mode + '_zeroshot_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[self.mode + '_zeroshot_mean_recall'][k] = sum_recall / float(self.num_rel)
        return


class SGNGMeanZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGNGMeanZeroShotRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_ng_zeroshot_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[self.mode + '_ng_zeroshot_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)],
                                                           50: [[] for i in range(self.num_rel)],
                                                           100: [[] for i in range(self.num_rel)]}
        self.result_dict[self.mode + '_ng_zeroshot_mean_recall_list'] = {20: [], 50: [], 100: []}

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        zeroshot_triplets = zeroshot_triplets - 1

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_ng_zeroshot_mean_recall'].items():
            result_str += '  zmR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero-shot Mean Recall.' % self.mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[self.mode + '_ng_zeroshot_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']
        if len(self.zeroshot_idx) > 0:
            for k in self.result_dict[self.mode + '_ng_zeroshot_mean_recall_collect']:
                # the following code are copied from Neural-MOTIFS
                match = reduce(np.union1d, pred_to_gt[:k])
                # NOTE: by kaihua, calculate Mean Recall for each category independently
                # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
                recall_hit = [0] * self.num_rel
                recall_count = [0] * self.num_rel
                for idx in range(gt_rels.shape[0]):
                    if idx in self.zeroshot_idx:
                        local_label = gt_rels[idx, 2]
                        recall_count[int(local_label)] += 1
                        recall_count[0] += 1

                for idx in range(len(match)):
                    matched_idx = int(match[idx])
                    # check if matched gt is the zero-shot
                    if matched_idx in self.zeroshot_idx:
                        local_label = gt_rels[matched_idx, 2]
                        recall_hit[int(local_label)] += 1
                        recall_hit[0] += 1

                for n in range(self.num_rel):
                    if recall_count[n] > 0:
                        self.result_dict[self.mode + '_ng_zeroshot_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self):
        for k, v in self.result_dict[self.mode + '_ng_zeroshot_mean_recall'].items():
            sum_recall = 0
            # we do not have 'bg' class
            for idx in range(self.num_rel):
                if len(self.result_dict[self.mode + '_ng_zeroshot_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[self.mode + '_ng_zeroshot_mean_recall_collect'][k][idx])
                self.result_dict[self.mode + '_ng_zeroshot_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[self.mode + '_ng_zeroshot_mean_recall'][k] = sum_recall / float(self.num_rel)
        return

"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""


class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[self.mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit / a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % self.mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        # we can get empty relationship, mostly due to center point collision
        if local_container['pred_rel_inds'].shape[0]:
            pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
            gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
            self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0
        else:
            self.pred_pair_in_gt = np.empty((0, 2), dtype=np.bool)

    def calculate_recall(self, global_container, local_container):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[self.mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
            # for sgcls and predcls
            if self.mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[self.mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[self.mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGMeanRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[self.mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[self.mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)],
                                                           50: [[] for i in range(self.num_rel)],
                                                           100: [[] for i in range(self.num_rel)]}
        self.result_dict[self.mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % self.mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[self.mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[self.mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[self.mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self):
        for k, v in self.result_dict[self.mode + '_mean_recall'].items():
            sum_recall = 0
            # we do not have 'bg' class
            for idx in range(self.num_rel):
                if len(self.result_dict[self.mode + '_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[self.mode + '_mean_recall_collect'][k][idx])
                self.result_dict[self.mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[self.mode + '_mean_recall'][k] = sum_recall / float(self.num_rel)
        return


"""
No Graph Constraint Mean Recall
"""


class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGNGMeanRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_ng_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[self.mode + '_ng_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)],
                                                              50: [[] for i in range(self.num_rel)],
                                                              100: [[] for i in range(self.num_rel)]}
        self.result_dict[self.mode + '_ng_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_ng_mean_recall'].items():
            result_str += 'ng-mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Mean Recall.' % self.mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[self.mode + '_ng_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[self.mode + '_ng_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[self.mode + '_ng_mean_recall_collect'][k][n].append(
                        float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self):
        for k, v in self.result_dict[self.mode + '_ng_mean_recall'].items():
            sum_recall = 0
            for idx in range(self.num_rel):
                if len(self.result_dict[self.mode + '_ng_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[self.mode + '_ng_mean_recall_collect'][k][idx])
                self.result_dict[self.mode + '_ng_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[self.mode + '_ng_mean_recall'][k] = sum_recall / float(self.num_rel)
        return


"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""


class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGAccumulateRecall, self).__init__(*args, **kwargs)

    def register_container(self):
        self.result_dict[self.mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[self.mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % self.mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self):
        for k, v in self.result_dict[self.mode + '_accumulate_recall'].items():
            self.result_dict[self.mode + '_accumulate_recall'][k] = float(
                self.result_dict[self.mode + '_recall_hit'][k][0]) / float(
                self.result_dict[self.mode + '_recall_count'][k][0] + 1e-10)

        return


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_iou(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_iou(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_iou(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def bbox_iou(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """

    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    iou = inter / (area1[:, None] + area2 - inter)

    return iou

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

