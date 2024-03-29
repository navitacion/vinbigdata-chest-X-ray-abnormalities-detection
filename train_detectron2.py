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
    wandb.config.update(dict(cfg.aug_kwargs_detection))
    wandb.config.update(dict(cfg.classification_kwargs))

    # omegaconf -> dict
    rep_aug_kwargs = OmegaConf.to_container(cfg.aug_kwargs_detection)

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
    img_size = cfg.data.img_size
    backbone = cfg.data.backbone
    use_class14 = cfg.data.use_class14

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if use_class14:
        class_name_dict.update({14: 'No finding'})

    # Register Dataset  --------------------------------------------------
    anno_df = pd.read_csv(os.path.join(data_dir, 'train_wbf_th0.7.csv'))

    if cfg.data.use_class14:
        pass
    else:
        anno_df = anno_df[anno_df['class_id'] != 14].reset_index(drop=True)

    # Extract rad id
    if cfg.data.rad_id != 'all':
        anno_df = anno_df[anno_df['rad_id'].isin(cfg.data.rad_id)].reset_index()

    if debug:
        anno_df = anno_df.head(100)

    # Split train, valid data - random
    if 'valid' in cfg.data.split_method:
        split_rate = float(cfg.data.split_method.split('_')[1]) / 100
        unique_image_ids = anno_df['image_id'].values
        unique_image_ids = np.random.RandomState(cfg.data.seed).permutation(unique_image_ids)
        train_image_ids = unique_image_ids[:int(len(unique_image_ids) * (1 - split_rate))]
        valid_image_ids = unique_image_ids[int(len(unique_image_ids) * (1 - split_rate)):]
        DatasetCatalog.register("xray_valid", lambda d='valid': get_xray_dict(anno_df, data_dir, cfg, valid_image_ids))
        MetadataCatalog.get("xray_valid").set(thing_classes=list(class_name_dict.values()))

    else:
        train_image_ids = anno_df['image_id'].values
    DatasetCatalog.register("xray_train", lambda d='train': get_xray_dict(anno_df, data_dir, cfg, train_image_ids))
    MetadataCatalog.get("xray_train").set(thing_classes=list(class_name_dict.values()))

    DatasetCatalog.register("xray_test", lambda d='test': get_test_xray_dict(data_dir))
    MetadataCatalog.get("xray_test").set(thing_classes=list(class_name_dict.values()))

    # Config  --------------------------------------------------
    detectron2_cfg = get_cfg()
    detectron2_cfg.aug_kwargs = CN(rep_aug_kwargs)
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.DATASETS.TRAIN = ("xray_train",)
    if 'valid' in cfg.data.split_method:
        detectron2_cfg.DATASETS.TEST = ("xray_valid",)
        detectron2_cfg.TEST.EVAL_PERIOD = cfg.train.max_iter // 10
    else:
        detectron2_cfg.DATASETS.TEST = ()
    detectron2_cfg.INPUT.MIN_SIZE_TRAIN = (img_size,)
    detectron2_cfg.INPUT.MAX_SIZE_TRAIN = img_size
    detectron2_cfg.DATALOADER.NUM_WORKERS = cfg.train.num_workers
    detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
    detectron2_cfg.SOLVER.IMS_PER_BATCH = cfg.train.ims_per_batch
    detectron2_cfg.SOLVER.BASE_LR = cfg.train.lr
    detectron2_cfg.SOLVER.MAX_ITER = cfg.train.max_iter
    detectron2_cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    detectron2_cfg.SOLVER.WARMUP_ITERS = 2000
    detectron2_cfg.SOLVER.CHECKPOINT_PERIOD = 200000
    detectron2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.train.batch_size_per_image
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
    detectron2_cfg.OUTPUT_DIR = output_dir
    detectron2_cfg.SEED = cfg.data.seed
    detectron2_cfg.PIXEL_MEAN = [103.530, 116.280, 123.675]
    detectron2_cfg.PIXEL_STD = [1.0, 1.0, 1.0]

    # Train  --------------------------------------------------
    os.makedirs(detectron2_cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(detectron2_cfg)
    trainer = MyTrainer(detectron2_cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Rename Last Weight
    renamed_model = f"{backbone.split('.')[0].replace('/', '-')}.pth"
    os.rename(os.path.join(cfg.data.output_dir, 'model_final.pth'),
              os.path.join(cfg.data.output_dir, renamed_model))

    # Logging
    for model_path in glob.glob(os.path.join(cfg.data.output_dir, '*.pth')):
        wandb.save(model_path)

    # Inference Setting  ------------------------------------------------------
    detectron2_cfg = get_cfg()
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
    detectron2_cfg.MODEL.WEIGHTS = os.path.join(output_dir, renamed_model)  # path to the model we just trained
    detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.data.score_th   # set a custom testing threshold

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