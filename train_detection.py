import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
import wandb
from sklearn.model_selection import StratifiedKFold

from torch import nn, optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.utils import seed_everything
from src.transform import ImageTransform
from src.lightning import ChestXrayDataModule, XrayLightningDetection
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
    wandb.login()
    wandb_logger = WandbLogger(project='VinBigData-Detection', reinit=True)
    wandb_logger.log_hyperparams(dict(cfg.data))
    wandb_logger.log_hyperparams(dict(cfg.train))
    wandb_logger.log_hyperparams(dict(cfg.aug_kwargs_classification))

    # Data Module  -------------------------------------------------------------------
    transform = ImageTransform(cfg, type='detection')
    cv = StratifiedKFold(n_splits=cfg.data.n_splits)
    dm = ChestXrayDataModule(data_dir, cfg, transform, cv, data_type='detection', sample=False)

    # Model  -----------------------------------------------------------
    net = get_effdet_model(cfg, pretrained=True, task='train')

    # Optimizer, Scheduler  -----------------------------------------------------------
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)
    # Lightning Module
    model = XrayLightningDetection(net, cfg, optimizer, scheduler)

    # Trainer  --------------------------------------------------------------------------
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=100,
        max_epochs=cfg.train.epoch,
        gpus=-1,
        num_sanity_val_steps=0,
        deterministic=True,
        amp_level='O2',
        amp_backend='apex'
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Stop Logging
    wandb.finish()

    for p in model.weight_paths:
        os.remove(p)


if __name__ == '__main__':
    main()
