# Run the training process
# This is essentially the same as running main.py but in a notebook cell

import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use']

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300

from dataloader import GEN_DATA_LISTS, ISICDataset
from data_utils import collate, pallet_isic
from model import SegNext
from losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer, Evaluator, ModelUtils
from gray2color import gray2color

# Use ISIC dataset palette for visualization
palette = pallet_isic
g2c = lambda x: gray2color(x, use_pallet='custom', custom_pallet=palette)

# Get data paths
data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
train_paths, val_paths, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()
data_lists.get_filecounts()

# Create datasets and dataloaders
train_data = ISICDataset(
    train_paths[0], train_paths[1], 
    config['img_height'], config['img_width'],
    config['Augment_data'], config['Normalize_data']
)

# Configure DataLoader
dataloader_kwargs = {
    'batch_size': config['batch_size'],
    'shuffle': True,
    'num_workers': config['num_workers'],
    'collate_fn': collate,
    'pin_memory': config['pin_memory']
}

train_loader = DataLoader(train_data, **dataloader_kwargs)

val_data = ISICDataset(
    val_paths[0], val_paths[1], 
    config['img_height'], config['img_width'],
    False, config['Normalize_data']
)

val_loader = DataLoader(val_data, **dataloader_kwargs)

# Initialize the model
model = SegNext(
    num_classes=config['num_classes'], 
    in_channnels=3, 
    embed_dims=[32, 64, 460, 256],
    ffn_ratios=[4, 4, 4, 4], 
    depths=[3, 3, 5, 2], 
    num_stages=4,
    dec_outChannels=256, 
    drop_path=float(config['stochastic_drop_path']),
    config=config
)

# Set device and move model to it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Set up the loss function, optimizer and learning rate scheduler
loss = FocalLoss()
criterion = lambda x, y: loss(x, y)
optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': config['learning_rate']}], weight_decay=0.0005)
scheduler = LR_Scheduler(
    config['lr_schedule'], 
    config['learning_rate'], 
    config['epochs'],
    iters_per_epoch=len(train_loader), 
    warmup_epochs=config['warmup_epochs']
)

# Initialize metrics and utilities
metric = ConfusionMatrix(config['num_classes'])
mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.load_chkpt(model, optimizer)

# Create trainer and evaluator
trainer = Trainer(model, config['batch_size'], optimizer, criterion, metric)
evaluator = Evaluator(model, metric)

# Training loop
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou = []

for epoch in range(config['epochs']):
    # Training phase
    pbar = tqdm(train_loader)
    model.train()
    ta, tl = [], []
    
    for step, data_batch in enumerate(pbar):
        scheduler(optimizer, step, epoch)
        loss_value = trainer.training_step(data_batch)
        iou = trainer.get_scores()
        trainer.reset_metric()
        
        tl.append(loss_value)
        ta.append(iou['iou_mean'])
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
    
    print(f'=> Average loss: {np.nanmean(tl)}, Average IoU: {np.nanmean(ta)}')

    # Validation phase (every 2 epochs)
    if (epoch + 1) % 2 == 0:
        model.eval()
        va = []
        vbar = tqdm(val_loader)
        
        for step, val_batch in enumerate(vbar):
            with torch.no_grad():
                evaluator.eval_step(val_batch)
                viou = evaluator.get_scores()
                evaluator.reset_metric()

            va.append(viou['iou_mean'])
            vbar.set_description(f'Validation - v_mIOU {viou["iou_mean"]:.4f}')

        # Get sample prediction for visualization
        img, gt, pred = evaluator.get_sample_prediction()
        tiled = np.hstack([img, g2c(gt)*255, g2c(pred)*255])
        
        # Display visualization
        plt.figure(figsize=(15, 5))
        plt.imshow(tiled)
        plt.title(f"Epoch {epoch+1} - Image, Ground Truth, Prediction")
        plt.axis('off')
        plt.show()
        
        # Update best IoU
        avg_viou = np.nanmean(va)
        total_avg_viou.append(avg_viou)
        curr_viou = np.nanmax(total_avg_viou)
        print(f'=> Averaged Validation IoU: {avg_viou:.4f}')

    # Save checkpoint if validation IoU improved
    if curr_viou > best_iou:
        best_iou = curr_viou
        mu.save_chkpt(model, optimizer, epoch, loss_value, best_iou)

print(f"Training completed. Best validation IoU: {best_iou:.4f}") 