import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import hydra

from src.transform import ImageTransform_detection
from src.dataset import ChestXrayDataset
from src.utils import display_bbox_image


def get_image(data_dir, idx, axes, transform, target_img_id=None):
    # Ground Truth
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train = train[train['rad_id'] == 'R9'].reset_index(drop=True)
    train_img_paths = [os.path.join(data_dir, 'train', f'{p}.png') for p in train['image_id']]
    target_image_id = train.iloc[idx]['image_id']
    img_path = os.path.join(data_dir, 'train', f'{target_image_id}.png')

    bboxes = train[train['image_id'] == target_image_id][['x_min', 'y_min', 'x_max', 'y_max']].values
    labels = train[train['image_id'] == target_image_id]['class_id'].values
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    display_bbox_image(img, bboxes, labels, ax=axes[0])
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Prediction
    dataset = ChestXrayDataset(train_img_paths, transform, df=train)
    transformed_img, target, image_id = dataset.__getitem__(idx)
    bboxes = target['boxes'].cpu().numpy()
    label = target['labels'].cpu().numpy()

    transformed_img = transformed_img.permute(1, 2, 0).cpu().numpy()

    display_bbox_image(transformed_img, bboxes, label, ax=axes[1])
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    del bboxes, labels

@hydra.main('config.yaml')
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input/original_png'
    transform = ImageTransform_detection(cfg)

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(28, 18))
    idx = 80

    get_image(data_dir, idx, axes, transform)
    plt.tight_layout()

    fig.savefig('Check_aug.jpg')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    main()
