from dataclasses import dataclass, field
from typing import Dict

import torch
import os
import hydra
from omegaconf import DictConfig
import glob
import pandas as pd
from torch.utils.data import DataLoader

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, create_model_from_config
from effdet.efficientdet import HeadNet

from src.dataset import ChestXrayDataset
from src.transform import ImageTransform_detection


def collate_fn(batch):
    return tuple(zip(*batch))


@hydra.main(config_name = "config.yaml")
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    # config = get_efficientdet_config('tf_efficientdet_d5')
    # net = EfficientDet(config, pretrained_backbone=False)
    #
    # config.num_classes = 1
    # config.image_size=512
    # net.class_net = HeadNet(config, num_outputs=config.num_classes)
    #
    # net = DetBenchTrain(net, config)

    from effdet import create_model
    net = create_model(
        'tf_efficientdet_d0',
        bench_task='train',
        num_classes=14,
        bench_labeler=True)


    data_dir = './input/original_png'
    transform = ImageTransform_detection(cfg)
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df = df[df['class_id'] != 14].reset_index(drop=True)
    img_paths = [os.path.join(data_dir, 'train', f'{p}.png') for p in df['image_id']]

    dataset = ChestXrayDataset(img_paths, transform, phase='train', df=df, data_type='detection')

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    img, targets, img_id = next(iter(dataloader))

    img = torch.stack(img)

    targets = [{k: v for k, v in t.items()} for t in targets]

    targets2 = {}
    targets2["bbox"] = [
        target["boxes"].float() for target in targets
    ]  # variable number of instances, so the entire structure can be forced to tensor
    targets2["cls"] = [target["labels"].float() for target in targets]


    # target_res = {}
    # boxes = [target['boxes'].float() for target in targets]
    # labels = [target['labels'].float() for target in targets]
    # target_res['bbox'] = boxes
    # target_res['cls'] = labels
    # print(img.size())
    # print(boxes)

    print(len(targets2["bbox"]))
    print(len(targets2["cls"]))

    loss = net(img, targets2)

    print(loss)


if __name__ == '__main__':
    main()

