import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


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
        if self.phase == 'test':
            img, image_id = self.load_image_for_cls(idx, phase=self.phase)
            return img, image_id

        elif self.data_type == 'classification':
            img, label, image_id = self.load_image_for_cls(idx, phase=self.phase)
            return img, label, image_id

        elif self.data_type == 'detection':
            image, boxes, labels, image_id = self.load_image_and_boxes(idx)

            # if random.random() > 0.33:
            #     image, boxes, labels, image_id = self.load_image_and_boxes(idx)
            # elif random.random() > 0.5:
            #     image, boxes, labels, image_id = self.load_cutmix_image_and_boxes(idx)
            # else:
            #     image, boxes, labels, image_id = self.load_mixup_image_and_boxes(idx)

            ## To prevent ValueError: y_max is less than or equal to y_min for bbox from albumentations bbox_utils
            labels = np.array(labels, dtype=np.int).reshape(len(labels), 1)
            combined = np.hstack((boxes.astype(np.int), labels))
            combined = combined[np.logical_and(combined[:,2] > combined[:,0],
                                               combined[:,3] > combined[:,1])]
            boxes = combined[:, :4]
            labels = combined[:, 4].tolist()

            target = {}
            target['boxes'] = boxes
            target['labels'] = torch.tensor(labels)
            target['image_id'] = torch.tensor([idx])
            print(image.shape)
            if self.transform:
                for i in range(10):
                    sample = self.transform(**{
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': labels
                    })
                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                        target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  ## ymin, xmin, ymax, xmax
                        break

                ## Handling case where no valid bboxes are present
                if len(target['boxes'])==0 or i==9:
                    return None
                else:
                    ## Handling case where augmentation and tensor conversion yields no valid annotations
                    try:
                        assert torch.is_tensor(image), f"Invalid image type:{type(image)}"
                        assert torch.is_tensor(target['boxes']), f"Invalid target type:{type(target['boxes'])}"
                    except Exception as E:
                        print("Image skipped:", E)
                        return None

            return image, target, image_id



        # image_id = os.path.basename(img_path).split('.')[0]
        #
        # # Load Image
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        # height, width = img.shape[:2]
        #
        # if not self.phase == 'test':
        #     # Get label
        #     target_df = self.df[self.df['image_id'] == image_id]
        #
        #     if self.data_type == 'detection':
        #         # bboxes
        #         bboxes = target_df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        #         # class_labels
        #         class_labels = target_df['class_id'].values.tolist()
        #
        #         # Augmentations
        #         img, bboxes, label = self.transform(img, bboxes, class_labels, self.phase)
        #
        #         # has bboxes
        #         if len(bboxes) != 0:
        #             bboxes = np.array(bboxes)
        #         # has no bboxes
        #         else:
        #             bboxes = np.array([[0, 0, height, width]])
        #             label = [14]
        #
        #         target = {}
        #         # Need yxyx format for EfficientDet.
        #         target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*bboxes)))).permute(1, 0)
        #         target['labels'] = torch.tensor(label, dtype=torch.float)
        #         target['image_id'] = torch.tensor([idx])
        #
        #         return img, target, image_id
        #
        #     elif self.data_type == 'classification':
        #         # Class_id == 14
        #         if target_df['class_id'].max() == 14:
        #             label = 1
        #         else:
        #             label = 0
        #
        #         # Augmentations
        #         img = self.transform(img, phase=self.phase)
        #         img = img.float()
        #
        #         return img, label, image_id
        #
        # # phase test  only image transform
        # else:
        #     width, height, _ = img.shape
        #     if self.transform is not None:
        #         img = self.transform(img, phase=self.phase)
        #     else:
        #         img = torch.from_numpy(img.transpose((2, 0, 1)))
        #         img = img / 255.
        #
        #     return img, image_id



    def load_image_for_cls(self, idx, phase):
        img_path = self.img_path[idx]
        image_id = os.path.basename(img_path).split('.')[0]

        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if phase != 'test':
            # Get label
            target_df = self.df[self.df['image_id'] == image_id]

            # Class_id == 14
            if target_df['class_id'].max() == 14:
                label = 1
            else:
                label = 0

            # Augmentations
            img = self.transform(img, phase=self.phase)
            img = img.float()

            return img, label, image_id

        else:
            if self.transform is not None:
                img = self.transform(img, phase=self.phase)
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1)))
                img = img / 255.

            return img, image_id


    def load_image_and_boxes(self, idx):
        img_path = self.img_path[idx]
        image_id = os.path.basename(img_path).split('.')[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records['class_id'].tolist()
        resize_transform = A.Compose([A.Resize(height=512, width=512, p=1.0)],
                                     p=1.0,
                                     bbox_params=A.BboxParams(
                                         format='pascal_voc',
                                         min_area=0.1,
                                         min_visibility=0.1,
                                         label_fields=['labels'])
                                     )

        resized = resize_transform(**{
            'image': image,
            'bboxes': boxes,
            'labels': labels
        })

        resized_bboxes = np.vstack((list(bx) for bx in resized['bboxes']))
        return resized['image'], resized_bboxes, resized['labels'], image_id

    def load_mixup_image_and_boxes(self, index):
        image, boxes, labels, image_id = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels, image_id = self.load_image_and_boxes(random.randint(0, len(self.img_path) - 1))
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels, r_labels)), image_id

    def load_cutmix_image_and_boxes(self, index, imsize=512):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.img_path) - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = np.array([], dtype=np.int)

        for i, index in enumerate(indexes):
            image, boxes, labels, image_id = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels = np.concatenate((result_labels, labels))

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
        result_boxes = result_boxes[index_to_use]
        result_labels = result_labels[index_to_use]

        return result_image, result_boxes, result_labels, image_id
