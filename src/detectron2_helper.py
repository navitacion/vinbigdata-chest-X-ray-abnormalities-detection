import os, json, cv2, random, glob, time, logging, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
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



def get_xray_dict(data_dir, use_class14=True):
    anno_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    if use_class14:
        pass
    else:
        anno_df = anno_df[anno_df['class_id'] != 14].reset_index(drop=True)

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
        if use_class14:
            bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']]
            bboxes['x_min'] = bboxes['x_min'].fillna(0)
            bboxes['y_min'] = bboxes['y_min'].fillna(0)
            bboxes['x_max'] = bboxes['x_max'].fillna(width)
            bboxes['y_max'] = bboxes['y_max'].fillna(height)
            bboxes = bboxes.values.tolist()
        else:
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


def get_submission(dataset_dicts, test_img_info_df, predictor, img_size):
    img_id_list = []
    sub_list = []

    for d in tqdm(dataset_dicts, total=len(dataset_dicts)):
        im = cv2.imread(d["file_name"])
        image_id = os.path.basename(d["file_name"]).split('.')[0]
        # Get Original Height, Width
        tmp = test_img_info_df[test_img_info_df['image_id'] == image_id]
        height = tmp['height'].values
        width = tmp['width'].values

        outputs = predictor(im)
        fields = outputs['instances'].get_fields()
        # Predict
        bboxes = fields['pred_boxes'].tensor.detach().cpu().numpy()
        scores = fields['scores'].detach().cpu().numpy()
        labels = fields['pred_classes'].detach().cpu().numpy()

        # Transform original shape
        bboxes = bboxes / img_size
        bboxes = bboxes * np.array([width, height, width, height]).reshape((1, 4))
        bboxes = bboxes.astype(int)

        # Get PredictionString
        sub_text = ''
        for j in range(bboxes.shape[0]):
            # class_id = 14 is NoData
            if labels[j] == 14:
                continue
            sub_text += f'{labels[j]} '
            sub_text += f'{scores[j]} '
            sub_text += ' '.join(map(str, list(bboxes[j])))
            sub_text += ' '

        if sub_text == '':
            sub_text = '14 1 0 0 1 1'

        img_id_list.append(image_id)
        sub_list.append(sub_text)

    sub = pd.DataFrame({
        'image_id': img_id_list,
        'PredictionString': sub_list
    })

    return sub