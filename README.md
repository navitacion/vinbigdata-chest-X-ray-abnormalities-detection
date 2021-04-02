# VinBigData Chest X-ray Abnormalites Detection


Kaggle Competition Repogitory


## References

- [Detectron2](https://github.com/facebookresearch/detectron2)

- [Detectron2 Backbone](https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py)

- [VinBigData detectron2 train](https://www.kaggle.com/corochann/vinbigdata-detectron2-train)

- [Weights & Biases](https://wandb.ai/site)

- [Windows10 + WSL2 + DockerDesktop + docker-compose +GPU(Nvidia)+Jupyterlabの環境構築](https://qiita.com/gonzou122/items/7b5e74d7c4c5f3e969af)


## Result

- Private Score: 0.236
- Rank: 200th (16%)


## Getting Started

Easy to do, only type command.

```commandline
docker-compose up --build -d
docker exec -it vin_env bash
```

## Solution

2 Classification (normality or abnormality) Model + Detection Model (Detectron2)

- 2 Classification Model
    - backbone: tf_efficientnet_b4_ns (from timm)
    - image_size: 384, 512
    - learning_rate: 0.0001
    - Epochs: 10
    - Folds: 5 (Stratified)
    

- Detection Model
    - backbone: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
    - image_size: 1024
    - learning_rate: 0.001
    - max_iter: 70000
    
Inference is performed using the following logic.

- Load image
- Use the model to infer whether there is an anomaly.
- If there is an anomaly in the image, object detection is performed by Detection Model.
- If not, submit text is "14 0 0 0 1 1" (which means there is nothing).


## Model Training


To train Classification Model, execute the following command.

```commandline
python train_classification.py
```

To train Detection Model, execute the following command.

**IMPORTANT:** Before executing, you must put on Classification Model Weight to directory "./weights"


```commandline
python train_detectron2.py
```


## Helper Function

### resize_img.py

Script for generating a resized image from an original image.

In training the object detection model, we used this script to compress the images in advance.


### wbf.py

Script for execute Weighted Box Function.

Refer from [this code](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/208468).

In this case, we used a threshold value of "iou_thr = 0.7".


### train_detection.py

Script to run efficientdet with Pytorch Lightning.

**Warning:** This script is not complete, so running it will result in an error. I'm leaving it as a reminder.
