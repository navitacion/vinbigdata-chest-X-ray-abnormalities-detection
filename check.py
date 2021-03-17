import os
import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
import wandb
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.utils import seed_everything
from src.transform import ImageTransform
from src.lightning import ChestXrayDataModule, XrayLightningClassification
from src.model import get_effdet_model


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    print('VinBigData Training Detection')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = cfg.data.data_dir
    seed_everything(cfg.data.seed)

    load_dotenv('.env')
    # wandb.login()
    # wandb_logger = WandbLogger(project='VinBigData-Detection', reinit=True)
    # wandb_logger.log_hyperparams(dict(cfg.data))
    # wandb_logger.log_hyperparams(dict(cfg.train))
    # wandb_logger.log_hyperparams(dict(cfg.aug_kwargs_classification))

    # Data Module  -------------------------------------------------------------------
    transform = ImageTransform(cfg, type='detection')
    cv = StratifiedKFold(n_splits=cfg.data.n_splits)
    dm = ChestXrayDataModule(data_dir, cfg, transform, cv, data_type='detection', sample=None)
    dm.prepare_data()
    dm.setup()

    dataloader = dm.train_dataloader()

    # Model  -----------------------------------------------------------
    net = get_effdet_model(cfg, pretrained=False, task='train')

    images, targets, image_id = next(iter(dataloader))

    images = torch.stack(images).float()

    target_res = {}
    boxes = [target['boxes'].float() for target in targets]
    labels = [target['labels'].float() for target in targets]
    target_res['bbox'] = boxes
    target_res['cls'] = labels

    output = net(images, target_res)
    loss = output['loss']

    net.eval()
    target_res = {}
    boxes = [target['boxes'].float() for target in targets]
    labels = [target['labels'].float() for target in targets]
    target_res['bbox'] = boxes
    target_res['cls'] = labels
    target_res["img_scale"] = torch.tensor([1.0] * cfg.train.batch_size, dtype=torch.float)
    target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * cfg.train.batch_size, dtype=torch.float)

    output = net(images, target_res)
    loss = output['loss']
    det = output['detections']
    predictions = []
    for i in range(images.shape[0]):
        boxes = det[i].detach().cpu().numpy()[:, :4]
        scores = det[i].detach().cpu().numpy()[:, 4]
        labels = det[i].detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > 0)[0]
        boxes = boxes[indexes]
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
        predictions.append({
            'boxes': boxes[indexes],
            'scores': scores[indexes],
            'labels': labels[indexes]
        })

    # Optimizer, Scheduler  -----------------------------------------------------------
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)
    # # Lightning Module
    # model = XrayLightningClassification(net, cfg, criterion, optimizer, scheduler)
    #
    # # Trainer  --------------------------------------------------------------------------
    # trainer = Trainer(
    #     logger=wandb_logger,
    #     log_every_n_steps=100,
    #     max_epochs=cfg.train.epoch,
    #     gpus=-1,
    #     num_sanity_val_steps=0,
    #     deterministic=True,
    #     amp_level='O2',
    #     amp_backend='apex'
    # )
    #
    # # Train
    # trainer.fit(model, datamodule=dm)

    # Stop Logging
    # wandb.finish()
    #
    # for p in model.weight_paths:
    #     os.remove(p)


if __name__ == '__main__':
    main()
