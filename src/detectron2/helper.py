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
from detectron2.structures import BoxMode


from src.transform import ImageTransform_classification_test
from src.model import Timm_model



def get_xray_dict(data_dir, cfg):
    anno_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    if cfg.data.use_class14:
        pass
    else:
        anno_df = anno_df[anno_df['class_id'] != 14].reset_index(drop=True)

    # Extract rad id
    if cfg.data.rad_id != 'all':
        anno_df = anno_df[anno_df['rad_id'].isin(cfg.data.rad_id)].reset_index()

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
        if cfg.data.use_class14:
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


def get_submission_det(d, predictor):
    im = cv2.imread(d["file_name"])
    image_id = os.path.basename(d["file_name"]).split('.')[0]

    outputs = predictor(im)
    fields = outputs['instances'].get_fields()
    # Predict
    bboxes = fields['pred_boxes'].tensor.detach().cpu().numpy()
    scores = fields['scores'].detach().cpu().numpy()
    labels = fields['pred_classes'].detach().cpu().numpy()

    # Transform original shape
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

    return image_id, sub_text


def get_submission(dataset_dicts, cfg, experiment, predictor):
    img_id_list = []
    sub_list = []

    # Classification Model Setting
    nets = []
    if cfg.data.use_classification:
        experiment.log_asset_folder(cfg.classification_kwargs.weight_dir)
        weight_paths = glob.glob(os.path.join(cfg.classification_kwargs.weight_dir, '*.pth'))
        for weight_path in weight_paths:
            backbone = os.path.basename(weight_path).split('-')[0]
            net = Timm_model(backbone, out_dim=2)
            net.load_state_dict(torch.load(weight_path))
            nets.append(net.eval())

    for d in tqdm(dataset_dicts, total=len(dataset_dicts)):
        if cfg.data.use_classification:
            # Classification Phase
            img = cv2.imread(d['file_name'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = ImageTransform_classification_test(cfg)
            img = transform(img, phase='test')
            out = [torch.softmax(m(img.unsqueeze(0)), dim=1) for m in nets]
            out = torch.cat(out)
            out = torch.mean(out, 0)
            p1 = out[1].item()   # 1: class_id == 14  0: class_id != 14

            if p1 > cfg.classification_kwargs.upper_th:
                image_id = os.path.basename(d["file_name"]).split('.')[0]
                sub_text = '14 1 0 0 1 1'
            elif p1 > cfg.classification_kwargs.lower_th and p1 < cfg.classification_kwargs.upper_th:
                # Det Predict and add class14
                image_id, sub_text = get_submission_det(d, predictor)
                sub_text += f' 14 {p1} 0 0 1 1'
            else:
                # Only Det
                image_id, sub_text = get_submission_det(d, predictor)
        else:
            # Detection Phase
            image_id, sub_text = get_submission_det(d, predictor)

        img_id_list.append(image_id)
        sub_list.append(sub_text)

    sub = pd.DataFrame({
        'image_id': img_id_list,
        'PredictionString': sub_list
    })

    return sub


"""
Referenced:
- https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
- https://www.kaggle.com/dhiiyaur/detectron-2-compare-models-augmentation/#data
"""

class AlbumentationsMapper:
    """Mapper which uses `albumentations` augmentations"""
    def __init__(self, cfg, is_train: bool = True):
        aug_kwargs = cfg.aug_kwargs
        aug_list = [
        ]
        if is_train:
            aug_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])
        self.transform = A.Compose(
            aug_list, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
        )
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        # aug_input = T.AugInput(image)
        # transforms = self.augmentations(aug_input)
        # image = aug_input.image

        prev_anno = dataset_dict["annotations"]
        bboxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)
        # category_id = np.array([obj["category_id"] for obj in dataset_dict["annotations"]], dtype=np.int64)
        category_id = np.arange(len(dataset_dict["annotations"]))

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_id)
        image = transformed["image"]
        annos = []
        for i, j in enumerate(transformed["category_ids"]):
            d = prev_anno[j]
            d["bbox"] = transformed["bboxes"][i]
            annos.append(d)
        dataset_dict.pop("annotations", None)  # Remove unnecessary field.

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict