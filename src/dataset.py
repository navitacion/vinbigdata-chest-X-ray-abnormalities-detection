import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, img_path, transform=None, phase='train', df=None, data_type='detection'):
        self.img_path = img_path
        self.transform = transform
        self.phase = phase
        self.df = df
        self.data_type = data_type

    def __len__(self):
        return len(self.img_path)


    def __getitem__(self, idx):

        img_path = self.img_path[idx]
        image_id = os.path.basename(img_path).split('.')[0]

        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.phase != 'test':
            # Get label
            target_df = self.df[self.df['image_id'] == image_id]

            if self.data_type == 'detection':
                # bboxes
                bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
                # class_labels
                class_labels = target_df['class_id'].values.tolist()

                print(bboxes)
                print(len(bboxes))
                print(list(map(int, bboxes[0])))

                # Augmentations
                img, bboxes, label = self.transform(img, bboxes, class_labels, self.phase)

                # has bboxes
                if len(bboxes) != 0:
                    bboxes = np.array(bboxes)
                    area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                # has no bboxes
                else:
                    bboxes = np.array([[0, 0, 1, 1]])
                    area = [1]
                    label = [14]

                target = {}
                target['boxes'] = torch.from_numpy(bboxes)
                target['labels'] = torch.tensor(label, dtype=torch.int64)
                target['image_id'] = torch.tensor([idx])
                target['area'] = torch.as_tensor(area, dtype=torch.float32)
                target['iscrowd'] = torch.zeros((target_df.shape[0], ), dtype=torch.int64)

                return img, target, image_id

            elif self.data_type == 'classification':
                # Class_id == 14
                if target_df['class_id'].max() == 14:
                    label = 1
                else:
                    label = 0

                # Augmentations
                img = self.transform(img, phase=self.phase)

                return img, label, image_id

        # phase test  only image transform
        else:
            width, height, _ = img.shape
            if self.transform is not None:
                img = self.transform(img, phase=self.phase)
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1)))
                img = img / 255.

            return img, image_id, (height, width)
