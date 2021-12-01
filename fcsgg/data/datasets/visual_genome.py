"""
Contains codes for creating compatible Visual Genome Dataset for Detectron2.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Kaihua Tang https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import collections
import numpy as np
from PIL import Image
import logging
import os, json, cv2, random, h5py, datetime
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.data.detection_utils import read_image
BOX_SCALE = 1024  # Scale at which we have the boxes
logger = logging.getLogger(__name__)

def convert_to_vg_dict(dataset_name):
    """
    Convert an Visual Genome dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids, useful when multiple datasets are used for training
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    predicate_categories = [
        {"id": id, "name": name}
        for id, name in enumerate(metadata.predicate_classes)
    ]

    logger.info("Converting dataset dicts into VG format")
    vg_images, vg_annotations = [], []

    for image_id, image_dict in enumerate(dataset_dicts):
        vg_image = {
            "image_id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        vg_images.append(vg_image)

        anns_per_image = image_dict.get("annotations", [])
        bboxes, cat_ids = [], []
        for annotation in anns_per_image:
            # For VG, we use XYXY_ABS
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
            bbox = [round(float(x), 3) for x in bbox]
            bboxes.append(bbox)
            cat_ids.append(reverse_id_mapper(annotation["category_id"]))
        vg_annotations.append({"image_id": image_dict.get("image_id", image_id),
                              "gt_boxes": bboxes,
                              "gt_classes": cat_ids,
                              "gt_rels": image_dict.get("relations", [])})

        # vg_annotations.append(vg_dict)

    logger.info(
        "Conversion finished, "
        f"#images: {len(vg_images)}, #annotations: {len(vg_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated VG json file for Detectron2.",
    }
    dataset_dict = {"info": info, "images": vg_images, "categories": categories,
                 "predicate_categories": predicate_categories, "licenses": None}
    if len(vg_annotations) > 0:
        dataset_dict["annotations"] = vg_annotations
    return dataset_dict


def convert_to_vg_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to VG format ...)")
            vg_dict = convert_to_vg_dict(dataset_name)

            logger.info(f"Caching VG format annotations at '{output_file}' ...")
            with PathManager.open(output_file, "w") as f:
                json.dump(vg_dict, f)

class VisualGenomeParser(object):
    def __init__(self, img_dir, roidb_file, dict_file, image_file,
                 transforms=None,
                 flip_aug=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        # assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        # self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file

        # self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = \
            self.load_info()  # contiguous 151, 51 containing __background__

        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.all_filenames, self.all_img_info = self.load_image_filenames()  # length equals to split_mask
        # dataset statistics
        self.predicate_stats = [] # will change in `load_graphs`


    def load_image_filenames(self, find_mismatch=False):
        """
        Loads the image filenames from visual genome from the JSON file that contains them.
        This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
        Parameters:
            image_file: JSON file. Elements contain the param "image_id".
            img_dir: directory where the VisualGenome images are located
        Return:
            List of filenames corresponding to the good images
        """
        with open(self.image_file, 'r') as f:
            im_data = json.load(f)
        corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
        fns = []
        img_info = []
        for i, img in enumerate(im_data):
            basename = '{}.jpg'.format(img['image_id'])
            if basename in corrupted_ims:
                continue

            filename = os.path.join(self.img_dir, basename)
            if os.path.exists(filename):
                if find_mismatch:
                    # 713545.jpg has exif info that changes the image
                    image = read_image(filename, "RGB")
                    h, w, _ = image.shape
                    if img['width'] != w or img['height'] != h:
                        im_data[i]['width'] = w
                        im_data[i]['height'] = h
                        print("mismatch on {}".format(filename))
                fns.append(filename)
                img_info.append(img)

        if len(fns) != 108073 or len(img_info) != 108073:
            logger.warning("No. of images available: {}. Please download the dataset following the instructions.".format(len(fns)))
        if find_mismatch:
            with open(self.image_file, 'w') as outfile:
                json.dump(im_data, outfile)
        return fns, img_info

    def load_info(self, add_bg=False):
        """
        Loads the file containing the visual genome label meanings
        """
        info = json.load(open(self.dict_file, 'r'))
        if add_bg:
            info['label_to_idx']['__background__'] = 0
            info['predicate_to_idx']['__background__'] = 0
            info['attribute_to_idx']['__background__'] = 0

        class_to_ind = info['label_to_idx']
        predicate_to_ind = info['predicate_to_idx']
        attribute_to_ind = info['attribute_to_idx']
        ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
        ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

        return ind_to_classes, ind_to_predicates, ind_to_attributes

    def load_graphs(self, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
        """
        Load the file containing the GT boxes and relations, as well as the dataset split
        Parameters:
            roidb_file: HDF5
            split: (train, val, or test)
            num_im: Number of images we want
            num_val_im: Number of validation images
            filter_empty_rels: (will be filtered otherwise.)
            filter_non_overlap: If training, filter images that dont overlap.
        Return:
            image_index: numpy array corresponding to the index of images we're using
            boxes: List where each element is a [num_gt, 4] array of ground
                        truth boxes (x1, y1, x2, y2)
            gt_classes: List where each element is a [num_gt] array of classes
            relationships: List where each element is a [num_r, 3] array of
                        (box_ind_1, box_ind_2, predicate) relationships
        """
        roi_h5 = h5py.File(self.roidb_file, 'r')
        data_split = roi_h5['split'][:]
        split_flag = 2 if split in {'test', 'minitest'} else 0
        split_mask = data_split == split_flag

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0

        image_index = np.where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if split == 'val':
                image_index = image_index[:num_val_im]
            elif split == 'train':
                image_index = image_index[num_val_im:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        # labels are in range [1, 150] !
        # minus one for class id to start from 0
        all_labels = roi_h5['labels'][:, 0] - 1
        all_attributes = roi_h5['attributes'][:, :]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        # labels are in range [1, 50] !
        # minus one for predicate id to start from 0
        _relation_predicates = roi_h5['predicates'][:, 0] - 1
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

        # Get everything by image.
        boxes = []
        gt_classes = []
        gt_attributes = []
        relationships = []
        for i in range(len(image_index)):
            i_obj_start = im_to_first_box[i]
            i_obj_end = im_to_last_box[i]
            i_rel_start = im_to_first_rel[i]
            i_rel_end = im_to_last_rel[i]

            boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
            gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

            if i_rel_start >= 0:
                predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
                obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
                assert np.all(obj_idx >= 0)
                assert np.all(obj_idx < boxes_i.shape[0])
                rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
            else:
                assert not filter_empty_rels
                rels = np.zeros((0, 3), dtype=np.int32)

            if filter_non_overlap:
                assert split == 'train'
                # construct BoxList object to apply boxlist_iou method
                # give a useless (height=0, width=0)
                boxes_i_obj = Boxes(boxes_i)
                inters = pairwise_iou(boxes_i_obj, boxes_i_obj)
                rel_overs = inters[rels[:, 0], rels[:, 1]]
                inc = np.where(rel_overs > 0.0)[0]

                if inc.size > 0:
                    rels = rels[inc]
                else:
                    split_mask[image_index[i]] = 0
                    continue

            boxes.append(boxes_i)
            gt_classes.append(gt_classes_i)
            gt_attributes.append(gt_attributes_i)
            relationships.append(rels)

        return split_mask, boxes, gt_classes, gt_attributes, relationships


    def get_things_classes(self):
        return self.ind_to_classes

    def get_predicate_classes(self):
        return self.ind_to_predicates

    def get_predicate_stats(self):
        roi_h5 = h5py.File(self.roidb_file, 'r')
        _relation_predicates = roi_h5['predicates'][:, 0] - 1
        self.predicate_stats = [v for k, v in sorted(collections.Counter(_relation_predicates).items())]
        return self.predicate_stats

    def get_frequency_bias(self):
        train_data = self.get_dicts("train",
                                      num_im=-1,
                                      num_val_im=0,
                                      filter_empty_rels=False,
                                      filter_non_overlap=False)
        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        freqs = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.float32)

        for data in train_data:
            gt_classes = [entry["category_id"] for entry in data["annotations"]]
            gt_relations = data["relations"]
            for i, gt_relation in enumerate(gt_relations):
                s, o, rel = gt_relation
                freqs[gt_classes[s], gt_classes[o], rel] += 1

        return freqs

    def get_dicts(self, split, num_im=-1, num_val_im=0, filter_empty_rels=True, filter_non_overlap=False, filter_duplicates=True):

        filter_non_overlap = filter_non_overlap and split in {'train', 'debug'}
        filter_duplicates = filter_duplicates and split in {'train', 'debug'}
        filter_empty_rels = True if split in {'debug', 'test', 'minitest'} else filter_empty_rels

        split_mask, gt_boxes, gt_classes, gt_attributes, relationships = self.load_graphs(
            split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=filter_non_overlap,
        )
        # filter by split
        filenames = [self.all_filenames[i] for i in np.where(split_mask)[0]]
        img_info = [self.all_img_info[i] for i in np.where(split_mask)[0]]

        dataset_dicts = []

        """
        To load a single image given the file name. Example:
        
            image_file_name = "datasets/vg/VG_100K/2364731.jpg"
            if image_file_name in filenames:
                image_idx = filenames.index(image_file_name)
                filenames = [image_file_name]
                img_info = [img_info[image_idx]]
                gt_boxes = [gt_boxes[image_idx]]
                gt_classes = [gt_classes[image_idx]]
                relationships = [relationships[image_idx]]
        """

        # iterate over each image and convert to dict
        for i, filename in enumerate(filenames):
            record = {}
            record["file_name"] = filename
            record["image_id"]  = img_info[i]['image_id']
            record["height"]    = img_info[i]['height']
            record["width"]     = img_info[i]['width']
            # box mode xyxy in the h5 file
            # important: recover original box from BOX_SCALE
            boxes = gt_boxes[i] / BOX_SCALE * max(record["width"], record["height"])
            # probably check if dim 0 is larger than zero
            assert boxes.shape[1] == 4
            category_ids = gt_classes[i]
            # deal with relations
            relations = relationships[i]
            if relations.shape[0] and filter_duplicates:
                # filter out non-unique relations (preserve same pair with different relations)
                relations = np.unique(relations, axis=0)
            # if we allow images without relations, it could be empty list
            relations = relations.tolist()
            record["relations"] = relations
            # for serialization
            record["annotations"] = [{
                "bbox": list(box),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(cat_id),
            } for box, cat_id in zip(boxes, category_ids)]
            dataset_dicts.append(record)

        return dataset_dicts


def register_visual_genome():
    """
    What we have here is the VG_stanford_filtered dataset.
    The file structure should be:

    fcsgg/
      |---- datasets/
         |---- vg/
            |---- VG-SGG.h5           # `roidb_file`, HDF5 containing the GT boxes, classes, and relationships
            |---- VG-SGG-dicts.json   # `dict_file`, JSON Contains mapping of classes/relationships to words
            |---- image_data.json     # `image_file`, HDF5 containing image filenames
            |---- VG_100K             # `img_dir`, contains all the images
               |---- 1.jpg
               |---- 2.jpg
               |---- ...

    """
    DATASET_BASE_DIR = "datasets/vg/"
    IMG_DIR = os.path.join(DATASET_BASE_DIR, "VG_100K")
    ROI_DB_FILE = os.path.join(DATASET_BASE_DIR, "VG-SGG-with-attri.h5")
    DICT_FILE = os.path.join(DATASET_BASE_DIR, "VG-SGG-dicts-with-attri.json")
    IMAGE_FILE = os.path.join(DATASET_BASE_DIR, "image_data.json")
    # change to False to disable computing frequency
    load_freq_bias = False
    # change to number > 0 for debug, -1 for all images
    num_images_to_load = {"train": -1,
                          "val": -1,
                          "test": -1,
                          "debug": 8,
                          "minitrain": 1000,
                          "minitest": 1000}
    vg_parser = VisualGenomeParser(IMG_DIR, ROI_DB_FILE, DICT_FILE, IMAGE_FILE)
    if load_freq_bias:
        freqs = vg_parser.get_frequency_bias()
    else:
        freqs = None
    for d in ["train", "val", "test", "debug", "minitrain", "minitest"]:
        DatasetCatalog.register("vg_" + d, lambda d=d: vg_parser.get_dicts(d, num_im=num_images_to_load[d],
                                                                           num_val_im=0,
                                                                           filter_empty_rels=False,
                                                                           filter_non_overlap=False))
        MetadataCatalog.get("vg_" + d).set(thing_classes=vg_parser.get_things_classes(),
                                           predicate_classes=vg_parser.get_predicate_classes(),
                                           predicate_stats=vg_parser.get_predicate_stats(),
                                           frequency_bias=freqs,
                                           evaluator_type="coco_vg")



register_visual_genome()

if __name__ == "__main__":
    annotations = DatasetCatalog.get("vg_test")