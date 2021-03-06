import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import albumentations as albu


train = pd.read_csv('./input/original_png/train.csv')
train_img_paths = glob.glob(os.path.join('./input/original_png/train/*.png'))
test_img_paths = glob.glob(os.path.join('./input/original_png/test/*.png'))

resize = 2048
output_train_img_dir = f'./input/resize_{resize}/train'
output_test_img_dir = f'./input/resize_{resize}/test'

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
    height, width, _ = img.shape
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

    # Original image shape
    res['height'] = height
    res['width'] = width

    # Concat bbox
    all_res = pd.concat([all_res, res], axis=0, ignore_index=True)


all_res.to_csv(f'./input/resize_{resize}/train.csv', index=False)


print('Resize Test Images')
img_id_list = []
height_list = []
width_list = []
# Test Image
for path in tqdm(test_img_paths):
    img = cv2.imread(path)
    height, width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_id = os.path.basename(path).split('.')[0]

    transform = albu.Compose([albu.Resize(width=resize, height=resize)], p=1.0)
    transformed = transform(image=img)
    img = transformed['image']

    # Save Image
    img = Image.fromarray(img)
    img.save(os.path.join(output_test_img_dir, f'{image_id}.png'))

    img_id_list.append(image_id)
    height_list.append(height)
    width_list.append(width)

all_res = pd.DataFrame({
    'image_id': img_id_list,
    'height': height_list,
    'width': width_list
})

all_res.to_csv(f'./input/resize_{resize}/test_image_info.csv', index=False)