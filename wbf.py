# Ref: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/208468

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from ensemble_boxes import *

# ===============================
# Default WBF config (you can change these)
iou_thr = 0.7
skip_box_thr = 0.0001
sigma = 0.1
# ===============================

# Loading the train DF
data_dir = './input/resize_2048'
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df.fillna(0, inplace=True)
df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0

results = []
image_ids = df["image_id"].unique()

for image_id in tqdm(image_ids, total=len(image_ids)):

    # All annotations for the current image.
    data = df[df["image_id"] == image_id]
    data = data.reset_index(drop=True)

    annotations = {}
    weights = []

    # WBF expects the coordinates in 0-1 range.
    max_value = data.iloc[:, 4:].values.max()
    data.loc[:, ["x_min", "y_min", "x_max", "y_max"]] = data.iloc[:, 4:] / max_value

    # Loop through all of the annotations
    for idx, row in data.iterrows():

        rad_id = row["rad_id"]

        if rad_id not in annotations:
            annotations[rad_id] = {
                "boxes_list": [],
                "scores_list": [],
                "labels_list": [],
            }

            # We consider all of the radiologists as equal.
            weights.append(1.0)

        annotations[rad_id]["boxes_list"].append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
        annotations[rad_id]["scores_list"].append(1.0)
        annotations[rad_id]["labels_list"].append(row["class_id"])

    boxes_list = []
    scores_list = []
    labels_list = []

    for annotator in annotations.keys():
        boxes_list.append(annotations[annotator]["boxes_list"])
        scores_list.append(annotations[annotator]["scores_list"])
        labels_list.append(annotations[annotator]["labels_list"])

    # Calculate WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    for idx, box in enumerate(boxes):
        results.append({
            "image_id": image_id,
            "class_id": int(labels[idx]),
            "rad_id": "wbf",
            "x_min": box[0] * max_value,
            "y_min": box[1] * max_value,
            "x_max": box[2] * max_value,
            "y_max": box[3] * max_value,
        })

results = pd.DataFrame(results)

results.to_csv(os.path.join(data_dir, f'train_wbf_th{iou_thr}.csv'), index=False)
