# FCSGG
A PyTorch implementation for the paper:

Fully Convolutional Scene Graph Generation \[[paper](https://arxiv.org/abs/2103.16083)\], CVPR 2021

Hengyue Liu<sup>*</sup>, Ning Yan, Masood Mortazavi, Bir Bhanu.  

<sup>*</sup> Work done in part as an intern at Futurewei Technologies Inc.

## Installation

The project is built upon [Detectron2](https://github.com/facebookresearch/detectron2). We incorporate Detectron2 as the submodule for easy use.

### Requirements

```
# clone this repo
git clone url fcsgg
cd fcsgg

# init and pull the Detectron2 submodule
git submodule init 
git submodule update
pip install -r requirements.txt
```

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.4 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


### Docker

If use docker, one can pull the latest pytorch image by

`docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel`

then mount the repo and run the container like

`docker run --gpus all --rm -ti --ipc=host --network=host --name="fcsgg" -v ~/fcsgg:/workspace/fcsgg pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel`

Maybe need extra libraries
```
apt install libgl1-mesa-glx

apt-get install libglib2.0-0
```

### Build Detectron2  

```shell script
# Detectron2 installation, or follow their instructions
python -m pip install -e detectron2 
```

## Dataset Preparation

1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. 

```
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P datasets/vg/
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P datasets/vg/
unzip -j datasets/vg/images.zip -d datasets/vg/VG_100K
unzip -j datasets/vg/images2.zip -d datasets/vg/VG_100K
```
Optionally, remove the .zip files.
```
rm datasets/vg/images.zip
rm datasets/vg/images2.zip
```   
  
2. Download the [scene graphs](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779871&authkey=AA33n7BRpB1xa3I) and extract them to `datasets/vg/VG-SGG-with-attri.h5`.

If use other paths, one may need to modify the related paths in file [visual_genome.py](fcsgg/data/datasets/visual_genome.py).

The correct structure of files should be

```
fcsgg/
      |---- datasets/
         |---- vg/
            |---- VG-SGG-with-attri.h5 # `roidb_file`, HDF5 containing the GT boxes, classes, and relationships
            |---- VG-SGG-dicts.json    # `dict_file`, JSON Contains mapping of classes/relationships to words
            |---- image_data.json      # `image_file`, HDF5 containing image filenames
            |---- VG_100K              # `img_dir`, contains all the images
               |---- 1.jpg
               |---- 2.jpg
               |---- ...

```

## Getting Started

For getting familiar with Detectron2, one can find useful material from [Detectron2 Doc](https://detectron2.readthedocs.io/index.html).

The program entry point is in `tools/train_net.py`. For a more instructional walk-through of the logic, I wrote a simple script in `tools/net_logic.py`, and for understanding of the Visual Genome dataset and dataloader, you can find some visualizations in [data_visualization.ipynb](tools/data_visualization.ipynb).

A minimum training example:
```
python tools/train_net.py \ 
--num-gpus 1 \
--config-file configs/quick_schedules/Quick-FCSGG-HRNet-W32.yaml
```
Detectron2 provides a key-value based config system that can be used to obtain standard, common behaviors. Common args can be found by running `python tools/train_net.py -h`. `--config-file` is required, other args like `--resume`(resume from the latest checkpoint) and `--eval-only`(only execute evaluation codes) are useful too.

By adding config arguments after `--config-file`, command line config overwrites the config in default `.yaml` file. For example, `MODEL.WEIGHTS ckpt_path` changes the checkpoint path.

A minimum evaluation example:
```
python tools/train_net.py \ 
--num-gpus 1 \
--eval-only \
--config-file configs/quick_schedules/Quick-FCSGG-HRNet-W32.yaml\
MODEL.WEIGHTS "put your .pth checkpoint path here"
```

## Benchmark

| Model | Checkpoint | Config |
| :---: | :---: | :---: |
| HRNetW32-1S   | [download](https://drive.google.com/file/d/1IpSDAZQW8irJyfWW5jep9fBVPRZcsTzl/view?usp=sharing) | [yaml](configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml) |
| ResNet50-4S-FPN×2   | [download](https://drive.google.com/file/d/10rtVr16RO2hd_JyiaBh8eNuxHvyTENR1/view?usp=sharing) | [yaml](configs/FCSGG-Res50-BiFPN-P2P5-MultiscaleHead-MS.yaml) |
| HRNetW48-5S-FPN×2  | [download](https://drive.google.com/file/d/1T7zZ_Rq5_mBhf1G89ab4w_SKz39IucsG/view?usp=sharing) | [yaml](configs/FCSGG_HRNet_W48_DualHRFPN_5s_Fixsize_640x1024_MS.yaml) |

## Citation
If you find our code or method helpful, please use the following BibTex entry.
```
@inproceedings{liu2021fully,
  title={Fully Convolutional Scene Graph Generation},
  author={Liu, Hengyue and Yan, Ning and Mortazavi, Masood and Bhanu, Bir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11546--11556},
  year={2021}
}
```


## Acknowledgments

We thanks these authors and implementations:

https://github.com/facebookresearch/detectron2

https://github.com/open-mmlab/mmdetection/blob/master

https://github.com/xingyizhou/CenterNet

https://github.com/FateScript/CenterNet-better

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/NiteshBharadwaj/part-affinity

