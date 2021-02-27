import os, json, cv2, random, glob, time, logging, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
import copy
import numpy as np

import torch
from detectron2.data import detection_utils as utils

from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm



def get_xray_dict(data_dir):
    anno_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    anno_df['x_min'] = anno_df['x_min'].fillna(0)
    anno_df['y_min'] = anno_df['y_min'].fillna(0)
    anno_df['x_max'] = anno_df['x_max'].fillna(1)
    anno_df['y_max'] = anno_df['y_max'].fillna(1)


    # anno_df = anno_df[anno_df['class_id'] != 14].reset_index(drop=True)

    dataset_dicts = []
    for i, img_id in enumerate(anno_df['image_id'].unique()):
        record = {}

        img_path = os.path.join(data_dir, 'train', f'{img_id}.png')
        height, width = cv2.imread(img_path).shape[:2]

        record['file_name'] = img_path
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        target_df = anno_df[anno_df['image_id'] == img_id]
        # bboxes
        bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        # class_labels
        class_labels = target_df['class_id'].values.tolist()

        objs = []
        for j in range(len(bboxes)):
            obj = {
                "bbox": list(map(int, bboxes[j])),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_labels[j]
            }
            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def get_test_xray_dict(data_dir):
    test_img_path = glob.glob(os.path.join(data_dir, 'test', f'*.png'))
    dataset_dicts = []
    for i, path in enumerate(test_img_path):
        record = {}

        height, width = cv2.imread(path).shape[:2]

        record['file_name'] = path
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        dataset_dicts.append(record)

    return dataset_dicts


