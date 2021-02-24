import os
import glob
import itertools
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import metrics

from src.dataset import ChestXrayDataset

def collate_fn(batch):
    return tuple(zip(*batch))


class ChestXrayDataModule(pl.LightningDataModule):
    """
    For Detection
    DataModule for VinBigData
    """
    def __init__(self, data_dir, cfg, transform, cv, sample=False, data_type='detection'):
        super(ChestXrayDataModule, self).__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        self.transform = transform
        self.cv = cv
        self.sample = sample
        self.data_type = data_type

    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        # Fillna
        self.df['x_min'] = self.df['x_min'].fillna(0)
        self.df['y_min'] = self.df['y_min'].fillna(0)
        self.df['x_max'] = self.df['x_max'].fillna(1)
        self.df['y_max'] = self.df['y_max'].fillna(1)

        self.df['is_anomary'] = self.df['class_id'].apply(lambda x: 1 if x == 14 else 0)
        if self.sample:
            self.df = self.df.sample(frac=self.sample, random_state=0)

        self.anomary_df = self.df[self.df['class_id'] != 14].reset_index()


    def setup(self, stage=None):
        if self.data_type == 'detection':
            df = self.anomary_df.copy()
            target_col = 'class_id'
        elif self.data_type == 'classification':
            df = self.df.copy()
            target_col = 'is_anomary'
        else:
            df, target_col = None, None

        # Validation
        df['fold'] = -1

        if self.data_type == 'classification':
            for i, (trn_idx, val_idx) in enumerate(self.cv.split(df, df[target_col])):
                df.loc[val_idx, 'fold'] = i
            fold = 0
            train = df[df['fold'] != fold].reset_index(drop=True)
            val = df[df['fold'] == fold].reset_index(drop=True)
            self.train_img_paths = [os.path.join(self.data_dir, 'train', f'{p}.png') for p in train['image_id']]
            self.val_img_paths = [os.path.join(self.data_dir, 'train', f'{p}.png') for p in val['image_id']]

        elif self.data_type == 'detection':
            unique_img_id = df['image_id'].unique()
            np.random.shuffle(unique_img_id)
            train_img_id = unique_img_id[:int(len(unique_img_id) * 0.8)]
            valid_img_id = unique_img_id[int(len(unique_img_id) * 0.8):]
            self.train_img_paths = [os.path.join(self.data_dir, 'train', f'{p}.png') for p in train_img_id]
            self.val_img_paths = [os.path.join(self.data_dir, 'train', f'{p}.png') for p in valid_img_id]

        # Image path
        self.test_img_paths = glob.glob(os.path.join(self.data_dir, 'test', '*.png'))

        # Dataset
        # Train
        self.train_dataset = ChestXrayDataset(self.train_img_paths,
                                              transform=self.transform,
                                              phase='train', df=df,
                                              data_type=self.data_type)
        # Valid
        self.valid_dataset = ChestXrayDataset(self.val_img_paths,
                                              transform=self.transform,
                                              phase='val', df=df,
                                              data_type=self.data_type)
        # Test
        self.test_dataset = ChestXrayDataset(self.test_img_paths,
                                             transform=self.transform,
                                             phase='test', df=None,
                                             data_type=self.data_type)

        self.collate_fn = collate_fn if self.data_type == 'detection' else None


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=False,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=False)


class XrayLightningClassification(pl.LightningModule):
    def __init__(self, net, cfg, criterion, optimizer, scheduler=None, experiment=None):
        """
        ------------------------------------
        Parameters
        net: torch.nn.Module
            Model
        cfg: DictConfig
            Config
        optimizer: torch.optim
            Optimizer
        scheduler: torch.optim.lr_scheduler
            Learning Rate Scheduler
        experiment: comet_ml.experiment
            Logger(Comet_ML)
        """
        super(XrayLightningClassification, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+9
        self.best_acc = 0
        self.epoch_num = 0
        self.acc_fn = metrics.Accuracy()
        self.train_step = 0
        self.valid_step = 0

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

    def forward(self, x):
        output = self.net(x)
        return output

    def step(self, batch):
        inp, label, image_id = batch
        inp = inp.float()
        label = label.long()

        out = self.forward(inp)
        loss = self.criterion(out, label)

        return loss, label, torch.sigmoid(out), image_id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id = self.step(batch)

        return {'val_loss': loss, 'logits': logits, 'labels': label, 'image_id': image_id}

    def test_step(self, batch, batch_idx):
        inp, image_id = batch
        inp = inp.type(torch.float32)
        out = self.forward(inp)
        logits = F.sigmoid(out)

        return {'preds': logits, 'image_id': image_id}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        # Accuracy
        acc = self.acc_fn(logits, labels.squeeze())

        # Logging
        if self.experiment is not None:
            logs = {'val/loss': avg_loss.item(), 'val/acc': acc.item()}
            # Logging
            self.experiment.log_metrics(logs, epoch=self.current_epoch)

        # Save Weights
        if self.best_loss > avg_loss or self.best_acc < acc:
            self.best_loss = min(avg_loss.item(), self.best_loss)
            self.best_acc = max(acc.item(), self.best_acc)
            logs = {'val/best_loss': self.best_loss, 'val/best_acc': self.best_acc}
            self.experiment.log_parameters(logs)

            expname = self.cfg.data.exp_name
            filename = f'{expname}_seed_{self.cfg.data.seed}_fold_{self.cfg.train.fold}_epoch_{self.epoch_num}_loss_{avg_loss.item():.3f}_acc_{acc.item():.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            if self.experiment is not None:
                self.experiment.log_model(name=filename, file_or_folder=filename)
                os.remove(filename)

        return {'avg_val_loss': avg_loss}


    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        preds = preds.detach().cpu().numpy()
        preds = pd.DataFrame(preds, columns=[f'label_{c}' for c in range(5)])
        # [tuple, tuple]
        img_ids = [x['image_id'] for x in outputs]
        # [list, list]
        img_ids = [list(x) for x in img_ids]
        img_ids = list(itertools.chain.from_iterable(img_ids))
        self.sub = preds
        self.sub.insert(0, 'image_id', img_ids)

        return None


class XrayLightningDetection(pl.LightningModule):
    def __init__(self, net, cfg, optimizer, scheduler=None, experiment=None):
        """
        ------------------------------------
        Parameters
        net: torch.nn.Module
            Model
        cfg: DictConfig
            Config
        optimizer: torch.optim
            Optimizer
        scheduler: torch.optim.lr_scheduler
            Learning Rate Scheduler
        experiment: comet_ml.experiment
            Logger(Comet_ML)
        """
        super(XrayLightningDetection, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+9
        self.best_acc = 0
        self.epoch_num = 0
        self.acc_fn = metrics.Accuracy()
        self.train_step = 0
        self.valid_step = 0

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

    def forward(self, x, targets):
        output = self.net(x, targets)
        return output

    def step(self, batch):
        inp, targets, image_id = batch
        loss_dict = self.forward(inp, targets)

        return loss_dict, image_id

    def training_step(self, batch, batch_idx):
        loss_dict, _ = self.step(batch)
        losses = sum(loss for loss in loss_dict.values())
        res = {'loss': losses}
        res.update(loss_dict)

        return res

    def validation_step(self, batch, batch_idx):
        self.net.train()
        with torch.no_grad():
            loss_dict, image_id = self.step(batch)
        losses = sum(loss for loss in loss_dict.values())
        res = {'val_loss': losses, 'image_id': image_id}
        res.update(loss_dict)

        return res

    def test_step(self, batch, batch_idx):
        self.net.eval()
        inp, image_id = batch
        out = self.net(inp)

        return {'outputs': out, 'image_id': image_id}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        loss_classifier = torch.stack([x['loss_classifier'] for x in outputs]).mean()
        loss_box_reg = torch.stack([x['loss_box_reg'] for x in outputs]).mean()
        loss_objectness = torch.stack([x['loss_objectness'] for x in outputs]).mean()
        loss_rpn_box_reg = torch.stack([x['loss_rpn_box_reg'] for x in outputs]).mean()

        logs = {
            'val/loss': avg_loss.item(),
            'val/loss_classifier': loss_classifier.item(),
            'val/loss_box_reg': loss_box_reg.item(),
            'val/loss_objectness': loss_objectness.item(),
            'val/loss_rpn_box_reg': loss_rpn_box_reg.item()
        }

        # Logging
        if self.experiment is not None:
            # Logging
            self.experiment.log_metrics(logs, epoch=self.current_epoch)

        # Save Weights
        if self.best_loss > avg_loss:
            self.best_loss = min(avg_loss.item(), self.best_loss)
            logs = {'val/best_loss': self.best_loss}
            self.experiment.log_parameters(logs)

            expname = self.cfg.data.exp_name
            filename = f'{expname}_seed_{self.cfg.data.seed}_fold_{self.cfg.train.fold}_epoch_{self.epoch_num}_loss_{avg_loss.item():.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            if self.experiment is not None:
                self.experiment.log_model(name=filename, file_or_folder=filename)
                os.remove(filename)

        return {'avg_val_loss': avg_loss}


    def test_epoch_end(self, outputs):

        img_id_list = []
        sub_list = []

        for output in outputs:
            image_ids = output['image_id']
            out = output['outputs']

            for i in range(len(image_ids)):
                image_id = image_ids[i]
                boxes = out[i]["boxes"].data.cpu().numpy().astype(np.int32)
                scores = out[i]["scores"].data.cpu().numpy()
                labels = out[i]["labels"].data.cpu().numpy()

                # score >= 0.5
                boxes = boxes[scores >= self.cfg.data.sub_th]
                labels = labels[scores >= self.cfg.data.sub_th]
                scores = scores[scores >= self.cfg.data.sub_th]

                sub_text = ''
                for j in range(boxes.shape[0]):
                    sub_text += f'{labels[j]} '
                    sub_text += f'{scores[j]} '
                    sub_text += ' '.join(map(str, list(boxes[j])))
                    sub_text += ' '

                if sub_text == '':
                    sub_text = '14 1 0 0 1 1'

                img_id_list.append(image_id)
                sub_list.append(sub_text)

        self.sub = pd.DataFrame({
            'image_id': img_id_list,
            'PredictionString': sub_list
        })

        filename = 'submission.csv'
        self.sub.to_csv(filename, index=False)
        self.experiment.log_asset(file_data=filename, file_name=filename)
        os.remove(filename)
