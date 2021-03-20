import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
from torch.utils.data import Dataset
import wandb

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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


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


def visualize(target_image_ids, data_dir, output_dir, predictor, score_th=0.0):
    for target_image_id in target_image_ids:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(28, 18))
        # Ground Truth
        img = cv2.imread(os.path.join(data_dir, 'train', f'{target_image_id}.png'))
        train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        bboxes = train[train['image_id'] == target_image_id][['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = train[train['image_id'] == target_image_id]['class_id'].values
        display_bbox_image(img, bboxes, labels, ax=axes[0])
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Predict
        outputs = predictor(img)
        fields = outputs['instances'].get_fields()
        bboxes = fields['pred_boxes'].tensor.detach().cpu().numpy()
        labels = fields['pred_classes'].detach().cpu().numpy()
        scores = fields['scores'].detach().cpu().numpy()

        bboxes = bboxes[scores > score_th]
        labels = labels[scores > score_th]

        display_bbox_image(img, bboxes, labels, ax=axes[1])
        axes[1].set_title('Predict')
        axes[1].axis('off')

        plt.tight_layout()

        filename = os.path.join(output_dir, f'result_{target_image_id}_th_{score_th}.jpg')
        fig.savefig(filename)
        wandb.save(filename)

        im = plt.imread(filename)
        wandb.log({"img": [wandb.Image(im, caption=f'result_{target_image_id}_th_{score_th}.jpg')]})
        del im

        plt.clf()
        plt.close()