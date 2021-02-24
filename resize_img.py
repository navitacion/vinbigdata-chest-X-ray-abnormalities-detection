import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import albumentations as albu


train = pd.read_csv('./input/train.csv')
train_img_paths = glob.glob(os.path.join('./input/train_png/*.png'))
test_img_paths = glob.glob(os.path.join('./input/test_png/*.png'))

resize = 512
output_train_img_dir = f'./input/train_{resize}'
output_test_img_dir = f'./input/test_{resize}'

os.makedirs(output_train_img_dir, exist_ok=True)
os.makedirs(output_test_img_dir, exist_ok=True)

transform = albu.Compose([albu.Resize(width=resize, height=resize)],
                         bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels']), p=1.0)

all_res = pd.DataFrame()

print('Resize Train Images')
# Train Image
for path in tqdm(train_img_paths):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_id = os.path.basename(path).split('.')[0]

    extracted = train[train['image_id'] == image_id]

    if extracted['class_id'].max() != 14:
        bbox = extracted[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        class_labels = extracted['class_id'].values.tolist()

        transform = albu.Compose([albu.Resize(width=resize, height=resize)],
                                 bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        transformed = transform(image=img, bboxes=bbox, class_labels=class_labels)

        img = transformed['image']
        bbox = transformed['bboxes']
        class_labels = transformed['class_labels']

        label = np.hstack((bbox, np.expand_dims(class_labels, axis=1)))
        res = pd.DataFrame(label, columns=['x_min', 'y_min', 'x_max','y_max', 'class_id'])
        res['image_id'] = image_id
        res['rad_id'] = extracted['rad_id'].values

    else:

        transform = albu.Compose([albu.Resize(width=resize, height=resize)], p=1.0)
        transformed = transform(image=img)

        img = transformed['image']

        res = extracted

    # Save Image
    img = Image.fromarray(img)
    img.save(os.path.join(output_train_img_dir, f'{image_id}.png'))

    # Concat bbox
    all_res = pd.concat([all_res, res], axis=0, ignore_index=True)


all_res.to_csv(f'./input/train_{resize}.csv', index=False)


print('Resize Test Images')
# Test Image
for path in tqdm(test_img_paths):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_id = os.path.basename(path).split('.')[0]

    transform = albu.Compose([albu.Resize(width=resize, height=resize)], p=1.0)
    transformed = transform(image=img)
    img = transformed['image']

    # Save Image
    img = Image.fromarray(img)
    img.save(os.path.join(output_test_img_dir, f'{image_id}.png'))