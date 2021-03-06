import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from comet_ml import Experiment
from sklearn.model_selection import StratifiedKFold


from torch import nn, optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer

from src.utils import seed_everything
from src.transform import ImageTransform_classification
from src.lightning import ChestXrayDataModule, XrayLightningClassification
from src.model import Timm_model


@hydra.main(config_name = "config.yaml")
def main(cfg: DictConfig):
    print('VinBigData Training Classification')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = cfg.data.data_dir
    seed_everything(cfg.data.seed)

    load_dotenv('.env')
    comet_api_key = os.environ['COMET_ML_KEY']
    comet_project_name = os.environ['COMET_ML_PROJECT_NAME']

    # Logging
    # Comet_ml
    experiment = Experiment(api_key=comet_api_key,
                            project_name=comet_project_name,
                            auto_param_logging=False,
                            auto_metric_logging=True,
                            parse_args=False,
                            auto_metric_step_rate=100)

    # Log Parameters
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))
    experiment.log_parameters(dict(cfg.aug_kwargs_classification))

    # Data Module  -------------------------------------------------------------------
    transform = ImageTransform_classification(cfg)
    cv = StratifiedKFold(n_splits=cfg.data.n_splits)
    dm = ChestXrayDataModule(data_dir, cfg, transform, cv, data_type='classification')

    # Model  -----------------------------------------------------------
    net = Timm_model(cfg.train.backbone, out_dim=2)
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Loss fn  -----------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # Optimizer, Scheduler  -----------------------------------------------------------
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)

    # Lightning Module
    model = XrayLightningClassification(net, cfg, criterion, optimizer, scheduler, experiment=experiment)

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


if __name__ == '__main__':
    main()