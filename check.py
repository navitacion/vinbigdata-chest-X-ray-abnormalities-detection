import os
import glob
import pandas as pd
import hydra
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.transform import ImageTransform_detection
from src.dataset import ChestXrayDataset
from src.model import get_faster_RCNN
from src.lightning import ChestXrayDataModule

@hydra.main('config.yml')
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input/resize_512'

    cv = StratifiedKFold(n_splits=cfg.data.n_splits)
    transform = ImageTransform_detection(img_size=512)
    dm = ChestXrayDataModule(data_dir, cfg, transform, cv, data_type='detection')
    dm.prepare_data()
    dm.setup()
    dataloader = dm.train_dataloader()

    batch = next(iter(dataloader))
    images, targets, image_ids = batch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


    # Model 1  -------------------------------------------------------
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)####True
    # model = model.to(device)
    #
    # ##注意　クラス数＋１
    # num_classes = 14
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Model 2  -------------------------------------------------------
    model = get_faster_RCNN(model_name='resnet50', pretrained=True, num_classes=14)


    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()

    print('#'*30)
    print('Result')
    print(loss_dict)

    print('#'*30)
    print(losses)

    print('#'*30)
    print(loss_value)

    # print('#'*30)
    # model.eval()
    # out = model(images)
    #
    # for i in range(len(out)):
    #     boxes = out[i]["boxes"].data.cpu().numpy()
    #     scores = out[i]["scores"].data.cpu().numpy()
    #     labels = out[i]["labels"].data.cpu().numpy()
    #     print(boxes)
    #     print(scores)
    #     print(labels)
    #     print('#'*30)


if __name__ == '__main__':
    main()
