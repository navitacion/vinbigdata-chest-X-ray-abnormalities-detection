import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
from torch.utils.data import Dataset

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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def display_bbox_image(img, bboxes, labels, ax):
    # Visualize
    ax.imshow(img)

    # bbox
    for xy, label in zip(bboxes, labels):
        if label == 14:
            continue
        x_min, y_min, x_max, y_max = map(int, xy)
        box = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min, ec='r', fill=False)
        ax.add_patch(box)
        ax.text(x=x_min, y=y_min, s=class_name_dict[label], weight='bold', c='r', va='bottom', size=20)

    return ax

# ref: https://albumentations.ai/docs/examples/example_bboxes/
def get_bbox_image(img, bboxes, labels):
    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White

    for xy, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = map(int, xy)
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BOX_COLOR, 2)

        ((text_width, text_height), _) = cv2.getTextSize(class_name_dict[label], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        img = cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        img = cv2.putText(
            img,
            text=class_name_dict[label],
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img