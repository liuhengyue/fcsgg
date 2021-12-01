"""
Scripts for testing the model inference and visualization of ground-truth data.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import unittest
import torch
import sys, os, platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch import build_model
from detectron2.checkpoint import DetectionCheckpointer
from fcsgg.config import add_fcsgg_config
from fcsgg.modeling.meta_arch import CenterNet
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from fcsgg.data.dataset_mapper import DatasetMapper

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
    # for mac os, change config to cpu
    if platform.system() == 'Darwin':
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.HRNET.NORM = "BN"
        cfg.MODEL.NECKS.NORM = "BN"
        cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.WEIGHTS = "output/vg/fcsgg_hrnet_w48_dualhrfpn_5s_fixsize_640x1024_ms/model_90k.pth"
    cfg.DATASETS.TRAIN = ("vg_debug",)
    cfg.DATASETS.TEST = ("vg_debug",)
    cfg.freeze()
    return cfg

def get_hook_fn(name):
    def hook_fn(self, input, output, name=name):
        print("{} output shape: {}".format(name, output.size()))
    return hook_fn

def setup_hook(model):
    for name, module in model.heads.named_modules():
        if name in ["cls_head", "wh_head", "reg_head", "raf_head"]:
            module.register_forward_hook(get_hook_fn(name))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    # unittest.main()
    cfg = setup()

    ###########    basic test    ###########

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    # print(model)
    # count_bins_parameters(model, convert=True)
    # print(count_parameters(model) / 1e6)
    model = model.eval()
    setup_hook(model)
    input = [{"image": torch.rand((3, 96, 512))}]
    output = model(input)

    ###########    save a model graph    ###########

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('log')  #
    # dummy_input = [{"image": torch.rand((3, 256, 256))}]
    # with SummaryWriter(comment='DLA-34') as w:
    #     w.add_graph(model, (dummy_input,))

    ###########    debug dataloader    ###########

    dataloader = build_detection_test_loader(cfg, "vg_minitest", mapper=DatasetMapper(cfg, True))
    dataset_metadata = MetadataCatalog.get("vg_debug")
    for d in dataloader:
        output = model(d)
        break

    ###########    debug predictor    ###########
    import random, cv2
    from detectron2.utils.visualizer import ColorMode
    from fcsgg.utils.visualizer import SceneGraphVisualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from fcsgg.data.datasets import register_visual_genome
    from detectron2.engine import DefaultPredictor
    import matplotlib
    import matplotlib.pyplot as plt
    predictor = DefaultPredictor(cfg)

    use_prediction = True
    dataset_dicts = DatasetCatalog.get("vg_debug")
    dataset_metadata = MetadataCatalog.get("vg_debug")
    # dataset_dicts = random.sample(dataset_dicts, 1)
    out_dir = "output/vg/quick_schedules/test/inference/vis"
    base_name = "results" if use_prediction else "gt"
    out_dir = os.path.join(out_dir, base_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, d in enumerate(dataset_dicts):
        im = cv2.imread(d["file_name"])
        image_basename = os.path.basename(d["file_name"]).split(".")[0]
        v = SceneGraphVisualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE
                       )
        if use_prediction:
            outputs = predictor(im) if use_prediction else d["annotations"]
            out, num_preds = v.draw_sg_predictions(outputs["scene_graph"].to("cpu"))
        else:
            out = v.draw_dataset_dict(d)

        # cv2.imwrite(os.path.join(out_dir, "{}.pdf".format(i)), out.get_image()[:, :, ::-1])
        if isinstance(out, list):
            for j, out_per_predicate in enumerate(out):
                out_per_predicate.save(os.path.join(out_dir, "{}_{}.pdf".format(i, j)))
                # plt.imshow(out_per_predicate.get_image())
                # plt.show()
        else:
            out.save(os.path.join(out_dir, "{}.pdf".format(image_basename)))
            # plt.imshow(out.get_image())
            # plt.show()


