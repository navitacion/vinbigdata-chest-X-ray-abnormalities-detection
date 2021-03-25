import os, cv2, glob
import pandas as pd
from tqdm import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from detectron2.structures import BoxMode


from src.transform import ImageTransform_cls_test
from src.model import Timm_model


def get_record(img_id, idx, df, data_dir, cfg):
    record = {}

    img_path = os.path.join(data_dir, 'train', f'{img_id}.png')
    height, width = cv2.imread(img_path).shape[:2]

    record['file_name'] = img_path
    record['image_id'] = idx
    record['height'] = height
    record['width'] = width

    target_df = df[df['image_id'] == img_id]
    # bboxes
    if cfg.data.use_class14:
        bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']]
        bboxes['x_min'] = bboxes['x_min'].fillna(0)
        bboxes['y_min'] = bboxes['y_min'].fillna(0)
        bboxes['x_max'] = bboxes['x_max'].fillna(width)
        bboxes['y_max'] = bboxes['y_max'].fillna(height)
        bboxes = bboxes.values.tolist()
    else:
        bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
    # class_labels
    class_labels = target_df['class_id'].values.tolist()

    objs = []
    for j in range(len(bboxes)):
        obj = {
            "bbox": list(map(int, bboxes[j])),
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": class_labels[j]
        }
        objs.append(obj)

    record['annotations'] = objs

    return record


def get_xray_dict(anno_df, data_dir, cfg, target_image_ids):
    df = anno_df[anno_df['image_id'].isin(target_image_ids)].reset_index(drop=True)

    dataset_dicts = []
    for idx, img_id in enumerate(df['image_id'].unique()):
        record = get_record(img_id, idx, df, data_dir, cfg)
        dataset_dicts.append(record)

    return dataset_dicts


def get_test_xray_dict(data_dir):
    test_img_path = glob.glob(os.path.join(data_dir, 'test', f'*.png'))
    dataset_dicts = []
    for i, path in enumerate(test_img_path):
        record = {}

        height, width = cv2.imread(path).shape[:2]

        record['file_name'] = path
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        dataset_dicts.append(record)

    return dataset_dicts


def get_predict_det(d, predictor, data_dir):
    im = cv2.imread(d["file_name"])
    resized_height, resized_width, _ = im.shape
    image_id = os.path.basename(d["file_name"]).split('.')[0]

    test_image_info = pd.read_csv(os.path.join(data_dir, 'test_image_info.csv'))
    row = test_image_info[test_image_info['image_id'] == image_id]
    org_height, org_width = row['height'].values[0], row['width'].values[0]

    h_ratio = org_height / resized_height
    w_ratio = org_width / resized_width

    outputs = predictor(im)
    fields = outputs['instances'].get_fields()
    # Predict
    bboxes = fields['pred_boxes'].tensor.detach().cpu().numpy()
    scores = fields['scores'].detach().cpu().numpy()
    labels = fields['pred_classes'].detach().cpu().numpy()

    # Transform original shape
    bboxes[:, [0, 2]] *= w_ratio
    bboxes[:, [1, 3]] *= h_ratio
    bboxes = bboxes.astype(int)

    # Get PredictionString
    sub_text = ''
    for j in range(bboxes.shape[0]):
        # class_id = 14 is NoData
        if labels[j] == 14:
            continue
        sub_text += f'{labels[j]} '
        sub_text += f'{scores[j]} '
        sub_text += ' '.join(map(str, list(bboxes[j])))
        sub_text += ' '

    if sub_text == '':
        sub_text = '14 1 0 0 1 1'

    return image_id, sub_text


def get_predict_classification(d, net, cfg, transform, device):
    # Classification Phase
    image_id = os.path.basename(d["file_name"]).split('.')[0]
    img_path = os.path.join(cfg.classification_kwargs.data_dir_cls, 'test', f'{image_id}.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # transform = ImageTransform_cls_test(cfg)
    img = transform(image=img)['image']
    img = img.to(device)

    p1 = torch.sigmoid(net(img.unsqueeze(0)))
    p1 = p1.detach().cpu().item()

    # Probability of class_id == 14
    return p1


def get_submission(dataset_dicts, cfg, predictor, device):
    img_id_list = []
    sub_list = []

    # Classification Model Setting
    nets = []
    transforms = []
    if cfg.data.use_classification:
        # experiment.log_asset_folder(cfg.classification_kwargs.weight_dir)
        wandb.save(os.path.join(cfg.classification_kwargs.weight_dir, '*.pth'))
        weight_paths = glob.glob(os.path.join(cfg.classification_kwargs.weight_dir, '*.pth'))
        for weight_path in weight_paths:
            backbone = os.path.basename(weight_path).split('-')[0]
            net = Timm_model(backbone, out_dim=1)
            net.load_state_dict(torch.load(weight_path))
            net = net.to(device)
            nets.append(net.eval())

            # モデルの入力に合わせた画像サイズで変換
            img_size = int(weight_path.split('ims_')[1].split('_')[0])
            transform = A.Compose([
                A.Resize(img_size, img_size, p=1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transforms.append(transform)

    for d in tqdm(dataset_dicts, total=len(dataset_dicts)):
        # Classification Phase
        p1 = 0
        for net, transform in zip(nets, transforms):
            p1 += get_predict_classification(d, net, cfg, transform, device)

        p1 /= len(nets)

        if p1 > cfg.classification_kwargs.upper_th:
            image_id = os.path.basename(d["file_name"]).split('.')[0]
            sub_text = '14 1 0 0 1 1'
        elif p1 > cfg.classification_kwargs.lower_th and p1 < cfg.classification_kwargs.upper_th:
            # Det Predict and add class14
            image_id, sub_text = get_predict_det(d, predictor, cfg.data.data_dir)
            sub_text += f'14 {p1} 0 0 1 1'
        else:
            # Only Det
            image_id, sub_text = get_predict_det(d, predictor, cfg.data.data_dir)
            sub_text += f'14 0 0 0 1 1'  # https://www.kaggle.com/prashantkikani/vinbigdata-post-processing

        img_id_list.append(image_id)
        sub_list.append(sub_text)

    sub = pd.DataFrame({
        'image_id': img_id_list,
        'PredictionString': sub_list
    })

    return sub
