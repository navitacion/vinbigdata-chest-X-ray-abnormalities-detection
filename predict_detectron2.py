import os, glob, hydra, cv2, shutil, time, datetime
from omegaconf import DictConfig
import wandb
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config.config import CfgNode as CN

from src.detectron2.helper import get_xray_dict, get_test_xray_dict, get_submission
from src.detectron2.trainer import MyTrainer
from src.utils import seed_everything
from src.utils import visualize


debug = False

@hydra.main(config_name = "config_detectron2")
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    seed_everything(cfg.data.seed)

    # wandb
    wandb.init(project='VinBigData-Detection')
    wandb.config.update(dict(cfg.data))
    wandb.config.update(dict(cfg.train))
    wandb.config.update(dict(cfg.classification_kwargs))

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

    # Setting  --------------------------------------------------
    data_dir = cfg.data.data_dir
    output_dir = cfg.data.output_dir
    backbone = cfg.data.backbone
    use_class14 = cfg.data.use_class14
    detectron2model_weight_path = 'COCO-Detection-faster_rcnn_R_50_FPN_3x_90.pth'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    if use_class14:
        class_name_dict.update({14: 'No finding'})

    # Register Dataset  --------------------------------------------------
    DatasetCatalog.register("xray_test", lambda d='test': get_test_xray_dict(data_dir))
    MetadataCatalog.get("xray_test").set(thing_classes=list(class_name_dict.values()))

    # Inference Setting  ------------------------------------------------------
    detectron2_cfg = get_cfg()
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
    detectron2_cfg.MODEL.WEIGHTS = detectron2model_weight_path  # path to the model we just trained
    detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.data.score_th   # set a custom testing threshold
    detectron2_cfg.OUTPUT_DIR = output_dir

    predictor = DefaultPredictor(detectron2_cfg)
    dataset_dicts = get_test_xray_dict(data_dir)

    # Visualize  ------------------------------------------------------
    target_image_ids = ['9a5094b2563a1ef3ff50dc5c7ff71345',
                        '22b8e616a61bbc4caaed0cf23b7159df',
                        '001d127bad87592efe45a5c7678f8b8d',
                        '008b3176a7248a0a189b5731ac8d2e95']

    for th in [0, 0.2, 0.5, 0.7]:
        visualize(target_image_ids, data_dir, output_dir, predictor, score_th=th)

    # Metrics
    if os.path.exists(os.path.join(output_dir, 'metrics.json')):
        metrics_df = pd.read_json(os.path.join(output_dir, 'metrics.json'), orient="records", lines=True)
        mdf = metrics_df.sort_values("iteration")

        mdf3 = mdf[~mdf["bbox/AP75"].isna()].reset_index(drop=True)
        for i in range(len(mdf3)):
            row = mdf3.iloc[i]
            wandb.log({'AP40': row["bbox/AP75"] / 100.})

        best_score = mdf3["bbox/AP75"].max() / 100.
        wandb.log({'Best-AP40-Score': best_score})

    # Inference  ------------------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sub = get_submission(dataset_dicts, cfg, predictor, device)

    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    now = now.strftime("%Y%m%d-%H%M%S")

    filename = f'submission_{now}.csv'
    sub.to_csv(os.path.join('./submission', filename), index=False)
    wandb.save(os.path.join('./submission', filename))
    time.sleep(30)

    wandb.finish()
    DatasetCatalog.clear()

if __name__ == '__main__':
    main()