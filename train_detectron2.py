import os, glob, hydra, random, cv2, shutil
from tqdm import tqdm
from dotenv import load_dotenv
from omegaconf import DictConfig
from comet_ml import Experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from src.detectron2_helper import get_xray_dict, get_test_xray_dict, get_submission
from src.utils import seed_everything, display_bbox_image

@hydra.main(config_name = "config_detectron2")
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
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
                            auto_metric_step_rate=100)

    # Log Parameters
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))

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

    if use_class14:
        class_name_dict.update({14: 'No finding'})

    # Register Dataset  --------------------------------------------------
    DatasetCatalog.register("xray_train", lambda d='train': get_xray_dict(data_dir))
    DatasetCatalog.register("xray_test", lambda d='test': get_test_xray_dict(data_dir))
    MetadataCatalog.get("xray_train").set(thing_classes=list(class_name_dict.values()))
    MetadataCatalog.get("xray_test").set(thing_classes=list(class_name_dict.values()))

    # Config  --------------------------------------------------
    detectron2_cfg = get_cfg()
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.DATASETS.TRAIN = ("xray_train",)
    detectron2_cfg.DATASETS.TEST = ()
    detectron2_cfg.INPUT.MIN_SIZE_TRAIN = (img_size,)
    detectron2_cfg.INPUT.MAX_SIZE_TRAIN = img_size
    detectron2_cfg.INPUT.RANDOM_FLIP = cfg.train.random_flip
    detectron2_cfg.DATALOADER.NUM_WORKERS = cfg.train.num_workers
    detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)  # Let training initialize from model zoo
    detectron2_cfg.SOLVER.IMS_PER_BATCH = cfg.train.ims_per_batch
    detectron2_cfg.SOLVER.BASE_LR = cfg.train.lr  # pick a good LR
    detectron2_cfg.SOLVER.MAX_ITER = cfg.train.max_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    detectron2_cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    detectron2_cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    detectron2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.train.batch_size_per_image   # faster, and good enough for this toy dataset (default: 512)
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    detectron2_cfg.OUTPUT_DIR = output_dir
    detectron2_cfg.SEED = cfg.data.seed

    # Train  --------------------------------------------------
    os.makedirs(detectron2_cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(detectron2_cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Logging
    for model_path in glob.glob(os.path.join(cfg.data.output_dir, '*.pth')):
        experiment.log_model(name=model_path, file_or_folder=model_path)

    experiment.log_table(os.path.join(output_dir, 'metrics.json'))

    # Inference Setting  ------------------------------------------------------
    model_name = 'model_final.pth'
    detectron2_cfg = get_cfg()
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
    detectron2_cfg.MODEL.WEIGHTS = os.path.join(output_dir, model_name)  # path to the model we just trained
    detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.predict.score_th   # set a custom testing threshold

    predictor = DefaultPredictor(detectron2_cfg)
    dataset_dicts = get_test_xray_dict(data_dir)

    # Visualize  ------------------------------------------------------
    target_image_ids = ['9a5094b2563a1ef3ff50dc5c7ff71345',
                        '22b8e616a61bbc4caaed0cf23b7159df',
                        '001d127bad87592efe45a5c7678f8b8d',
                        '008b3176a7248a0a189b5731ac8d2e95']

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
        display_bbox_image(img, bboxes, labels, ax=axes[1])
        axes[1].set_title('Predict')
        axes[1].axis('off')

        plt.tight_layout()

        filename = os.path.join(output_dir, f'result_{target_image_id}.jpg')
        fig.savefig(filename)
        experiment.log_image(filename)
        os.remove(filename)


    # Inference  ------------------------------------------------------
    test_img_info_df = pd.read_csv(os.path.join(data_dir, 'test_image_info.csv'))

    sub = get_submission(dataset_dicts, test_img_info_df, predictor, img_size)

    filename = os.path.join(output_dir, 'submission.csv')
    sub.to_csv(filename, index=False)
    experiment.log_asset(file_data=filename, file_name='submission.csv')

    shutil.rmtree(output_dir)

if __name__ == '__main__':
    main()