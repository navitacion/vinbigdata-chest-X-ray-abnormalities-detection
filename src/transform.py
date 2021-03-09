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
    def __init__(self, cfg):
        super(ImageTransform_classification, self).__init__()

        transform_train_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_classification.aug_train).items()]
        transform_train_list.append(ToTensorV2())
        transform_val_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_classification.aug_val).items()]
        transform_val_list.append(ToTensorV2())
        transform_test_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_classification.aug_test).items()]
        transform_test_list.append(ToTensorV2())

        self.transform = {
            'train': albu.Compose(transform_train_list, p=1.0),
            'val': albu.Compose(transform_val_list, p=1.0),
            'test': albu.Compose(transform_test_list, p=1.0)
        }


class ImageTransform_classification_test(BaseTransform):
    def __init__(self, cfg):
        super(ImageTransform_classification_test, self).__init__()

        transform_test_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_classification.aug_test).items()]
        transform_test_list.append(ToTensorV2())

        self.transform = {
            'test': albu.Compose(transform_test_list, p=1.0)
        }


class ImageTransform_detection(BaseTransform):
    def __init__(self, cfg):
        super(ImageTransform_detection, self).__init__()

        transform_train_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_detection.aug_train).items()]
        transform_train_list.append(ToTensorV2())
        transform_val_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_detection.aug_val).items()]
        transform_val_list.append(ToTensorV2())
        transform_test_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(cfg.aug_kwargs_detection.aug_test).items()]
        transform_test_list.append(ToTensorV2())

        bbox_params = albu.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_labels'])

        self.transform = {
            'train': albu.Compose(transform_train_list, p=1.0, bbox_params=bbox_params),
            'val': albu.Compose(transform_val_list, p=1.0, bbox_params=bbox_params),
            'test': albu.Compose(transform_test_list, p=1.0)
        }