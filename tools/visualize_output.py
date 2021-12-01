"""
Scripts for testing the model inference and visualization detection results.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import torch
import sys, os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch import build_model
from detectron2.checkpoint import DetectionCheckpointer
from fcsgg.config import add_fcsgg_config
from fcsgg.modeling.meta_arch import CenterNet
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from fcsgg.data.dataset_mapper import DatasetMapper
import random, cv2
from detectron2.utils.visualizer import ColorMode
from fcsgg.utils.visualizer import SceneGraphVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from fcsgg.data.datasets import register_visual_genome
from detectron2.engine import DefaultPredictor
import matplotlib
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("detectron2")

def setup():
    """
    Create configs return it.
    """
    config_file = "configs/FCSGG_HRNet_W48_DualHRFPN_5s_Fixsize_640x1024_MS.yaml"
    cfg = get_cfg()
    add_fcsgg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.MODEL.DEVICE = "cuda" #"cuda"
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = "output/vg/fcsgg_hrnet_w48_dualhrfpn_5s_fixsize_640x1024_ms/model_90k.pth"
    cfg.DATASETS.TEST = ("vg_minitest",)
    cfg.freeze()
    return cfg


def save_gt_graph(data, out_dir, object_class_names, predicate_class_names):
    gt_rels = data['scene_graph'].get_extra("gt_relations").cpu().numpy()
    gt_classes = data['scene_graph'].get_extra("gt_classes").cpu().numpy()
    object_labels = [object_class_names[i] for i in gt_classes]
    subj_ids = [int(gt_rel[0]) for gt_rel in gt_rels]
    subj_labels = [object_class_names[gt_classes[gt_rel[0]]] for gt_rel in gt_rels]
    obj_ids = [int(gt_rel[1]) for gt_rel in gt_rels]
    obj_labels = [object_class_names[gt_classes[gt_rel[1]]] for gt_rel in gt_rels]
    predicate_labels = [predicate_class_names[gt_rel[2]] for gt_rel in gt_rels]
    # .txt file line by line
    with open(os.path.join(out_dir, 'gt_graph.txt'), 'w') as f:
        for (subj_label, obj_label, predicate_label, gt_rel) in zip(subj_labels, obj_labels, predicate_labels, gt_rels):
            f.write('{}_{} {} {}_{} \n'.format(subj_label, gt_rel[0], predicate_label, obj_label, gt_rel[1]))

    # .json for GraphViz
    gt_dict = {
        "url": "../" + data["file_name"],
        "objects": [{"name": object_label} for object_label in object_labels],
        "attributes": [],
        "relationships": [{"predicate": predicate_label, "object": obj_id, "subject": subj_id}
                          for (predicate_label, obj_id, subj_id) in zip(predicate_labels, obj_ids, subj_ids)]
    }
    with open(os.path.join(out_dir, "gt_graph.json"), "w") as f:
        f.write(json.dumps(gt_dict, indent = 4))
    return



def visualize_detections(data, output, out_dir, object_class_names, predicate_class_names, image_format="pdf"):
    d = data[0]
    output = output[0]
    im = cv2.imread(d["file_name"])
    image_basename = os.path.basename(d["file_name"]).split(".")[0]
    v = SceneGraphVisualizer(im[:, :, ::-1],
                   metadata=dataset_metadata,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE
                   )
    outs, num_preds = v.draw_sg_predictions(output["scene_graph"].to("cpu"))

    image_out_dir = os.path.join(out_dir, str(num_preds) + "-" + image_basename)
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)
    for i, out in enumerate(outs):
        # plt.imshow(out.get_image())
        # plt.show()
        out.save(os.path.join(image_out_dir, "{}.{}".format(i, image_format)))

    save_gt_graph(d, image_out_dir, object_class_names, predicate_class_names)

if __name__ == '__main__':
    cfg = setup()

    ###########    build model   ###########

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model = model.eval()
    if cfg.MODEL.DEVICE == "cuda":
        model = model.cuda()

    out_dir = "output/vg/quick_schedules/test/inference/vis"
    base_name = "results"
    dataset_name = "vg_minitest" # vg_debug
    out_dir = os.path.join(out_dir, base_name, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataloader = build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))
    dataset_metadata = MetadataCatalog.get(dataset_name)
    object_class_names = dataset_metadata.thing_classes
    predicate_class_names = dataset_metadata.predicate_classes
    with torch.no_grad():
        for d in dataloader:
            output = model(d)
            visualize_detections(d, output, out_dir, object_class_names, predicate_class_names)
            # break





