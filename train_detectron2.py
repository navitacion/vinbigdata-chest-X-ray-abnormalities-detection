import os, glob, hydra, cv2, shutil
from dotenv import load_dotenv
from omegaconf import DictConfig
from comet_ml import Experiment

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config.config import CfgNode as CN

from src.detectron2.helper import get_xray_dict, get_test_xray_dict, get_submission
from src.detectron2.trainer import MyTrainer
from src.utils import seed_everything
from src.utils import visualize

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
    experiment.log_parameters(dict(cfg.aug_kwargs_detection))
    experiment.log_parameters(dict(cfg.classification_kwargs))

    # omegaconf -> dict
    rep_aug_kwargs = dict(cfg.aug_kwargs_detection)
    rep_aug_kwargs = {k: dict(v) for k, v in rep_aug_kwargs.items()}

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
    DatasetCatalog.register("xray_train", lambda d='train': get_xray_dict(data_dir, cfg))
    DatasetCatalog.register("xray_test", lambda d='test': get_test_xray_dict(data_dir))
    MetadataCatalog.get("xray_train").set(thing_classes=list(class_name_dict.values()))
    MetadataCatalog.get("xray_test").set(thing_classes=list(class_name_dict.values()))

    # Config  --------------------------------------------------
    detectron2_cfg = get_cfg()
    detectron2_cfg.aug_kwargs = CN(rep_aug_kwargs)
    detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone))
    detectron2_cfg.DATASETS.TRAIN = ("xray_train",)
    detectron2_cfg.DATASETS.TEST = ()
    detectron2_cfg.INPUT.MIN_SIZE_TRAIN = (img_size,)
    detectron2_cfg.INPUT.MAX_SIZE_TRAIN = img_size
    detectron2_cfg.DATALOADER.NUM_WORKERS = cfg.train.num_workers
    detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
    detectron2_cfg.SOLVER.IMS_PER_BATCH = cfg.train.ims_per_batch
    detectron2_cfg.SOLVER.BASE_LR = cfg.train.lr
    detectron2_cfg.SOLVER.MAX_ITER = cfg.train.max_iter
    detectron2_cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    detectron2_cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    detectron2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.train.batch_size_per_image
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
    detectron2_cfg.OUTPUT_DIR = output_dir
    detectron2_cfg.SEED = cfg.data.seed

    # Train  --------------------------------------------------
    os.makedirs(detectron2_cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(detectron2_cfg)
    trainer = MyTrainer(detectron2_cfg)
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
    detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.data.score_th   # set a custom testing threshold

    predictor = DefaultPredictor(detectron2_cfg)
    dataset_dicts = get_test_xray_dict(data_dir)

    # Visualize  ------------------------------------------------------
    target_image_ids = ['9a5094b2563a1ef3ff50dc5c7ff71345',
                        '22b8e616a61bbc4caaed0cf23b7159df',
                        '001d127bad87592efe45a5c7678f8b8d',
                        '008b3176a7248a0a189b5731ac8d2e95']

    visualize(target_image_ids, data_dir, output_dir, experiment, predictor)

    # Inference  ------------------------------------------------------
    sub = get_submission(dataset_dicts, cfg, experiment, predictor)

    filename = os.path.join(output_dir, 'submission.csv')
    sub.to_csv(filename, index=False)
    experiment.log_asset(file_data=filename, file_name='submission.csv')

    shutil.rmtree(output_dir)

if __name__ == '__main__':
    main()