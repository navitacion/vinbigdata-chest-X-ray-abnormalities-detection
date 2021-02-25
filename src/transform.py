from abc import ABCMeta
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        self.transform = None

    def __call__(self, img, bboxes=None, class_labels=None, phase='train'):
        if bboxes is not None and class_labels is not None:
            transformed = self.transform[phase](image=img, bboxes=bboxes, class_labels=class_labels)
            transformed_img = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            return transformed_img, transformed_bboxes, transformed_class_labels

        else:
            transformed = self.transform[phase](image=img)
            transformed_img = transformed['image']

            return transformed_img


class ImageTransform_classification(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_classification, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'test': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_detection(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_detection, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.Resize(int(img_size*1.5), int(img_size*1.5)),
                albu.RandomResizedCrop(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels'])),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels'])),

            'test': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }