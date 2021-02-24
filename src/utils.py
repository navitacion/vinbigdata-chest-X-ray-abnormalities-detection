import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
from torch.utils.data import Dataset


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
        x_min, y_min, x_max, y_max = xy
        box = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min, ec='r', fill=False)
        ax.add_patch(box)
        ax.text(x=x_min, y=y_min, s=label)

    return ax