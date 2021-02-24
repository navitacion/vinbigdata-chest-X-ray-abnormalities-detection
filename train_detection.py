import os
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from sklearn.model_selection import StratifiedKFold


from torch import nn, optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer

from src.utils import seed_everything
from src.transform import ImageTransform_classification, ImageTransform_detection
from src.lightning import ChestXrayDataModule, XrayLightningDetection
from src.model import get_faster_RCNN


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('VinBigData Training Objective Detection')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = './input/resize_512'
    seed_everything(cfg.data.seed)

    # Logging
    # Comet_ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name,
                            auto_param_logging=False,
                            auto_metric_logging=True,
                            auto_metric_step_rate=100)

    # Log Parameters
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))

    # Data Module  -------------------------------------------------------------------
    transform = ImageTransform_detection()
    cv = StratifiedKFold(n_splits=cfg.data.n_splits)
    dm = ChestXrayDataModule(data_dir, cfg, transform, cv, data_type='detection')

    # Model  -----------------------------------------------------------
    net = get_faster_RCNN(model_name='resnet50', pretrained=True, num_classes=14)
    # Log Model Graph
    experiment.set_model_graph(str(net))


    # Optimizer, Scheduler  -----------------------------------------------------------
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)

    # Lightning Module
    model = XrayLightningDetection(net, cfg, optimizer, scheduler, experiment=experiment)

    # Trainer  --------------------------------------------------------------------------
    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        gpus=-1,
        num_sanity_val_steps=0,
        amp_level='O2',
        amp_backend='apex'
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Test
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()