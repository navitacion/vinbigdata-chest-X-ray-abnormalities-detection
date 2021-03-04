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
    use_class14 = cfg.data.use_class14

    if use_class14:
        class_name_dict.update({14: 'No finding'})

    # Inference Setting  ------------------------------------------------------
    model_name = glob.glob(os.path.join('./detectron2_weights', '*.pth'))

    for w in model_name:
        backbone = w.split('-')[0]
        detectron2_cfg = get_cfg()
        detectron2_cfg.merge_from_file(model_zoo.get_config_file(backbone + '.yaml'))
        detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15 if use_class14 else 14
        detectron2_cfg.MODEL.WEIGHTS = os.path.join(output_dir, w)  # path to the model we just trained
        detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.data.score_th   # set a custom testing threshold

        predictor = DefaultPredictor(detectron2_cfg)
        dataset_dicts = get_test_xray_dict(data_dir)

        # Inference  ------------------------------------------------------
        sub = get_submission(dataset_dicts, cfg, experiment, predictor)

        filename = os.path.join(output_dir, f"submission_{backbone}.csv")
        sub.to_csv(filename, index=False)
        experiment.log_asset(file_data=filename, file_name=f"submission_{backbone}.csv")

if __name__ == '__main__':
    main()