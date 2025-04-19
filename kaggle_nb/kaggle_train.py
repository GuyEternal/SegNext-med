import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from isic_dataset import get_data_loaders, TrainAugmentation, ValAugmentation
from metrics import BinarySegmentationMetrics, bce_dice_loss
from tqdm import tqdm
import numpy as np
import random
import segmentation_models_pytorch as smp

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model on ISIC dataset')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3+'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34', 
                        help='Encoder for the segmentation model')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, 
                        help='Size to resize images to')
    parser.add_argument('--weights', type=str, default='imagenet', 
                        help='Initial weights for encoder')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='../input/isic-2018-task1-data',
                        help='Directory containing ISIC data')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory for saving outputs')
    return parser.parse_args()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_model(model_name, encoder_name, encoder_weights, in_channels, classes):
    """
    Create segmentation model based on architecture name
    """
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'manet':
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    metrics = BinarySegmentationMetrics()
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item())
        metrics.update(torch.sigmoid(outputs), masks)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 
                         'IoU': f'{metrics.get_iou():.4f}'})
    
    return loss_meter.avg, metrics.get_metrics()

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    loss_meter = AverageMeter()
    metrics = BinarySegmentationMetrics()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            loss_meter.update(loss.item())
            metrics.update(torch.sigmoid(outputs), masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 
                             'IoU': f'{metrics.get_iou():.4f}'})
    
    return loss_meter.avg, metrics.get_metrics()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define augmentations
    train_aug = TrainAugmentation(img_size=args.img_size)
    val_aug = ValAugmentation(img_size=args.img_size)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_aug,
        val_transform=val_aug
    )
    
    # Create model
    model = get_model(
        model_name=args.model,
        encoder_name=args.encoder,
        encoder_weights=args.weights,
        in_channels=3,
        classes=1
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['iou']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_{args.encoder}_best.pth"))
            print(f"Saved new best model with IoU: {best_iou:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
        }, os.path.join(args.output_dir, f"{args.model}_{args.encoder}_latest.pth"))
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main() 