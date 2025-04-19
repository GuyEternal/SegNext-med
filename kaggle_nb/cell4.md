# Update utils.py to properly handle the ISIC dataset
%%writefile utils.py
import math
import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import cv2, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (images_transform, masks_transform, torch_imgresizer,
                        torch_resizer)

class ModelUtils(object):
    def __init__(self, num_classes, chkpt_pth, exp_name):
        self.num_classes = num_classes
        self.chkpt_pth = chkpt_pth
        self.exp_name = exp_name
    
    def save_chkpt(self, model, optimizer, epoch=0, loss=0, iou=0):
        print('-> Saving checkpoint')
        torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'iou': iou,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(self.chkpt_pth, f'{self.exp_name}.pth'))

    def load_chkpt(self, model, optimizer=None):
        
        try:
            print('-> Loading checkpoint')
            chkpt = torch.load(os.path.join(self.chkpt_pth, f'{self.exp_name}.pth'))
            epoch = chkpt['epoch']
            loss = chkpt['loss']
            iou = chkpt['iou']
            model.load_state_dict(chkpt['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            print(f'[INFO] Loaded Model checkpoint: epoch={epoch} loss={loss} iou={iou}')
        except FileNotFoundError:
            print('[INFO] No checkpoint found')

class Trainer(object):
    def __init__(self, model, batch, optimizer, criterion, metric):
        self.model = model
        self.batch = batch
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def training_step(self, batched_data):
        # Handle differently based on dataset type (binary or multi-class)
        if config['num_classes'] == 2:  # ISIC binary segmentation
            # Stack images and convert to tensor
            img_batch = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() 
                                    for img in batched_data['img']], dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Stack labels and convert to tensor, keeping original size
            lbl_batch = torch.stack([torch.from_numpy(lbl).long() 
                                    for lbl in batched_data['lbl']], dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:  # Cityscapes multi-class segmentation (original approach)
            img_batch = images_transform(batched_data['img'])
            lbl_batch = torch_resizer(masks_transform(batched_data['lbl']))
        
        self.optimizer.zero_grad()

        preds = self.model.forward(img_batch)
        loss = self.criterion(preds, lbl_batch)

        loss.backward()
        self.optimizer.step()

        preds = preds.argmax(1)
        preds = preds.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, preds)

        return loss.item()

class Evaluator(object):
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def eval_step(self, data_batch):
        # Handle differently based on dataset type (binary or multi-class)
        if config['num_classes'] == 2:  # ISIC binary segmentation
            # Stack images and convert to tensor
            self.img_batch = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() 
                                    for img in data_batch['img']], dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Stack labels and convert to tensor, keeping original size
            lbl_batch = torch.stack([torch.from_numpy(lbl).long() 
                                    for lbl in data_batch['lbl']], dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:  # Cityscapes multi-class segmentation (original approach)
            self.img_batch = images_transform(data_batch['img'])
            lbl_batch = torch_resizer(masks_transform(data_batch['lbl']))
        
        with torch.no_grad():
            preds = self.model.forward(self.img_batch) # already softmaxed

        preds = preds.argmax(1)
        self.preds = preds.cpu().numpy()
        self.lbl_batch = lbl_batch.cpu().numpy()
        self.metric.update(self.lbl_batch, self.preds)
        
    def get_sample_prediction(self):
        # Check if we're working with ISIC binary segmentation
        if config['num_classes'] == 2:
            # Get the first image directly from NCHW format
            img = self.img_batch[0].cpu().permute(1, 2, 0).numpy()
            lbl = self.lbl_batch[0]
            pred = self.preds[0]
            return (img*255).astype(np.uint8), lbl.astype(np.uint8), pred.astype(np.uint8)
        else:
            # Original approach for Cityscapes
            self.img_batch = torch_imgresizer(self.img_batch).detach().cpu().numpy()
            img = np.transpose(self.img_batch[0,...], (1,2,0))
            lbl = self.lbl_batch[0,...]
            pred = self.preds[0,...]
            return (img*255).astype(np.uint8), lbl.astype(np.uint8), pred.astype(np.uint8) 