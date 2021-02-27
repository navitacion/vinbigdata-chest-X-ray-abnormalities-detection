import os, json, cv2, random, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

data_dir = './input/resize_512'
output_dir = './detectron2_output/exp01'
img_size = 512
model_name = 'model_0006999.pth'
score_th = 0.5
# Ref: https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py
backbone = "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"

# Register Dataset
DatasetCatalog.register("xray_test", lambda  d='test': get_test_xray_dict(data_dir))
MetadataCatalog.get("xray_test").set(thing_classes=list(class_name_dict.values()))


# Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(backbone))
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = output_dir
cfg.SEED = 42
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_th   # set a custom testing threshold

# Predict
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get('xray_test')
dataset_dicts = get_test_xray_dict(data_dir)


cols = 3
rows = 3
fig, axes = plt.subplots(rows, cols, figsize=(18, 18))

for d, ax in zip(random.sample(dataset_dicts, 9), axes.ravel()):
    img = cv2.imread(d["file_name"])
    img_id = os.path.basename(d["file_name"]).split('.')[0]
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    # cv2_imshow(out.get_image()[:, :, ::-1])
    #cv2.imwrite(str(outdir / f"vinbigdata{index}.jpg"), out.get_image()[:, :, ::-1])
    ax.imshow(out.get_image()[:, :, ::-1])
    ax.set_title(f"{img_id}")
    ax.axis('off')

fig.savefig(os.path.join(output_dir, 'result.jpg'))


# Inference
img_id_list = []
sub_list = []
test_img_info_df = pd.read_csv(os.path.join(data_dir, 'test_image_info.csv'))

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

sub.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)