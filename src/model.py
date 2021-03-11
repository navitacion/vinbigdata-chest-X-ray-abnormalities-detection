import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from timm import create_model
import timm


class Timm_model(nn.Module):
    def __init__(self, model_name, pretrained=True, out_dim=5, features_only=False):
        super(Timm_model, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained, features_only=features_only)

        if 'efficientnet' in model_name:
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=out_dim)
        elif 'vit' in model_name:
            self.base.head = nn.Linear(in_features=self.base.head.in_features, out_features=out_dim)
        elif "dm_nfnet" in model_name:
            self.base.head.fc = nn.Linear(in_features=self.base.head.fc.in_features, out_features=out_dim)
        else:
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=out_dim)

    def forward(self, x):
        return self.base(x)


# Detection Model  -----------------------------------------------------------------
class Timm_model_featuremap(nn.Module):
    def __init__(self, model_name, img_size, pretrained=True):
        super(Timm_model_featuremap, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained)
        # Out Channels
        z = torch.randn(2, 3, img_size, img_size)
        self.out_channels = self.base.forward_features(z).size(1)

    def forward(self, x):
        return self.base.forward_features(x)


# Ref: https://github.com/pytorch/vision/blob/10d5a55c332771164c13375f445331c52f8de6f1/torchvision/models/detection/faster_rcnn.py
def get_faster_RCNN(model_name, pretrained=True, num_classes=14, img_size=512):
    backbone = Timm_model_featuremap(model_name, img_size, pretrained=pretrained)

    anchor_generator = AnchorGenerator(sizes=((8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 512),),
                                       aspect_ratios=((0.2, 0.5, 0.7, 1.0, 1.5, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler
    )

    return model


# Pretrained on COCO
# Based on Resnet50
def get_original_faster_RCNN(num_classes=14,  pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)####True

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



if __name__ == '__main__':
    net = create_model('dm_nfnet_f4', pretrained=False)

    print(net)

    net = Timm_model("dm_nfnet_f4", pretrained=False)
    print(net)

    # Print Timm Models
    # model_names = timm.list_models(pretrained=True)
    # print(model_names)

    # print('#'*30)
    # print(backbone)