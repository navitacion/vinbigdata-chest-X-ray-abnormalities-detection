import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from timm import create_model


class Timm_model(nn.Module):
    def __init__(self, model_name, pretrained=True, out_dim=5, features_only=False):
        super(Timm_model, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained, features_only=features_only)

        if 'efficientnet' in model_name:
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=out_dim)
        elif 'vit' in model_name:
            self.base.head = nn.Linear(in_features=self.base.head.in_features, out_features=out_dim)
        else:
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=out_dim)

    def forward(self, x):
        return self.base(x)


# class Timm_model_featuremap(nn.Module):
#     def __init__(self, model_name, pretrained=True, features_only=False):
#         super(Timm_model_featuremap, self).__init__()
#         self.base = create_model(model_name, pretrained=pretrained, features_only=features_only)
#         # resnet50
#         self.out_channels = 2048
#
#     def forward(self, x):
#         return self.base(x)[-1]

class Timm_model_featuremap(nn.Module):
    def __init__(self, model_name, pretrained=True, features_only=False):
        super(Timm_model_featuremap, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained)
        # resnet50
        self.out_channels = 2048

    def forward(self, x):
        return self.base.forward_features(x)


# Ref: https://github.com/pytorch/vision/blob/10d5a55c332771164c13375f445331c52f8de6f1/torchvision/models/detection/faster_rcnn.py
def get_faster_RCNN(model_name, pretrained=True, num_classes=14):
    backbone = Timm_model_featuremap(model_name, pretrained=pretrained, features_only=True)

    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler
    )

    return model




if __name__ == '__main__':
    net = create_model('tf_efficientnet_b4_ns', pretrained=True, features_only=True)

    z = torch.randn(4, 3, 224, 224)

    out = net(z)[-1]
    print(out.size())

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    out = backbone(z)
    print(out.size())

    print(net)
    # print('#'*30)
    # print(backbone)