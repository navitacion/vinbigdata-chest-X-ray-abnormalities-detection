import os, json, cv2, random, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from src.detectron2_helper import get_xray_dict, get_test_xray_dict

class_name_dict = {
    0: 'Aortic enlargement',
    1: 'Atelectasis',
    2: 'Calcification',
    3: 'Cardiomegaly',
    4: 'Consolidation',
    5: 'ILD',
    6: 'Infiltration',
    7: 'Lung Opacity',
    8: 'Nodule/Mass',
    9: 'Other lesion',
    10: 'Pleural effusion',
    11: 'Pleural thickening',
    12: 'Pneumothorax',
    13: 'Pulmonary fibrosis',
}

# Setting
data_dir = './input/resize_512'
output_dir = './detectron2_output/exp02'
img_size = 512
# Ref: https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py
backbone = "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"
# backbone = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# Register Dataset
DatasetCatalog.register("xray_train", lambda d='train': get_xray_dict(data_dir))
MetadataCatalog.get("xray_train").set(thing_classes=list(class_name_dict.values()))

# Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(backbone))
cfg.DATASETS.TRAIN = ("xray_train",)
cfg.DATASETS.TEST = ()
cfg.INPUT.MIN_SIZE_TRAIN = (img_size,)
cfg.INPUT.MAX_SIZE_TRAIN = img_size
cfg.INPUT.RANDOM_FLIP = 'horizontal'
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = output_dir
cfg.SEED = 42

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()