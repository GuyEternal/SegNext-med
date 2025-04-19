#!/usr/bin/env python3
"""
Test script to validate the ISIC-2017 dataset pipeline with SegNext model
This script tests:
1. Data loading from small_isic_dataset
2. Model initialization with 2 classes
3. A few quick training iterations
4. Visualization of segmentation masks
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imgviz
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load configuration
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# Import necessary modules
from dataloader import GEN_DATA_LISTS, ISICDataset
from data_utils import collate, pallet_isic, encode_labels
from model import SegNext
from losses import FocalLoss
from metrics import ConfusionMatrix
from gray2color import gray2color

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def visualize_batch(batch, predictions=None):
    """Visualize a batch of images, labels and predictions"""
    s = 255
    img_list = []
    
    # Convert images and add to list
    for i in range(len(batch['img'])):
        img_list.append((batch['img'][i]*s).astype(np.uint8))
    
    # Convert binary masks to RGB directly with red for lesions
    for i in range(len(batch['lbl'])):
        # Create RGB mask (black background, red lesions)
        mask_vis = np.zeros((batch['lbl'][i].shape[0], batch['lbl'][i].shape[1], 3), dtype=np.uint8)
        # Set lesion pixels to red (255, 0, 0)
        mask_vis[batch['lbl'][i] == 1] = [255, 0, 0]
        img_list.append(mask_vis)
    
    # Add predictions if provided
    if predictions is not None:
        for i in range(predictions.shape[0]):
            # Create RGB prediction visualization (black background, red lesions)
            pred_vis = np.zeros((predictions[i].shape[0], predictions[i].shape[1], 3), dtype=np.uint8)
            # Set lesion pixels to red (255, 0, 0)
            pred_vis[predictions[i] == 1] = [255, 0, 0]
            img_list.append(pred_vis)
    
    # Create a grid of images
    rows = 2 if predictions is None else 3
    cols = len(batch['img'])
    grid = imgviz.tile(img_list, shape=(rows, cols), border=(255, 0, 0))
    
    plt.figure(figsize=(12, 4))
    plt.imshow(grid)
    plt.axis('off')
    plt.title('Top: Images, Middle: Ground Truth, Bottom: Predictions' if predictions is not None else 'Top: Images, Bottom: Ground Truth')
    plt.tight_layout()
    plt.savefig('isic_test_visualization.png', dpi=300)
    plt.close()
    
    print(f"Visualization saved to isic_test_visualization.png")

def test_data_loading():
    """Test if the ISIC dataset can be loaded successfully"""
    print("\n=== Testing Data Loading ===")
    
    # Override data_dir to ensure using the small_isic_dataset
    data_dir = "./small_isic_dataset"
    
    # Initialize data lists
    data_lists = GEN_DATA_LISTS(data_dir, config['sub_directories'])
    train_paths, val_paths, test_paths = data_lists.get_splits()
    classes = data_lists.get_classes()
    data_lists.get_filecounts()
    
    # Check if classes are correct for binary segmentation
    assert len(classes) == 2, f"Expected 2 classes, got {len(classes)}: {classes}"
    print(f"Classes: {classes}")
    
    # Create dataset and dataloader
    train_data = ISICDataset(
        train_paths[0], train_paths[1], 
        config['img_height'], config['img_width'],
        augment_data=False, normalize=True
    )
    
    # Check if dataset is not empty
    assert len(train_data) > 0, "Training dataset is empty"
    print(f"Training dataset size: {len(train_data)}")
    
    # Test batch loading
    batch_size = min(4, len(train_data))
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate
    )
    
    # Get a sample batch
    batch = next(iter(train_loader))
    
    # Check shapes
    print(f"Loaded batch with {len(batch['img'])} images")
    print(f"Image shape: {batch['img'][0].shape}")
    print(f"Label shape: {batch['lbl'][0].shape}")
    
    # Check if masks are binary (0 and 1 only)
    unique_values = set()
    for lbl in batch['lbl']:
        unique_values.update(np.unique(lbl))
    
    print(f"Unique mask values: {unique_values}")
    assert unique_values.issubset({0, 1}), f"Masks should only contain 0 and 1, found {unique_values}"
    
    # Visualize the batch
    visualize_batch(batch)
    
    return train_loader

def test_model_initialization():
    """Test if the model can be initialized with 2 classes"""
    print("\n=== Testing Model Initialization ===")
    
    # Initialize model
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
    
    model = model.to(device)
    
    # Check model's last layer for correct number of output channels
    # This depends on your specific model architecture
    print("Model initialized successfully")
    print(f"Model expects {config['num_classes']} classes")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model

def test_training_iteration(model, train_loader):
    """Test a few training iterations"""
    print("\n=== Testing Training Iterations ===")
    
    # Set model to training mode
    model.train()
    
    # Define loss and optimizer
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Set up metrics
    metric = ConfusionMatrix(config['num_classes'])
    
    # Run a few iterations
    max_iterations = 3
    
    for iteration, batch in enumerate(train_loader):
        if iteration >= max_iterations:
            break
        
        # Get images and labels
        images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in batch['img']], dim=0).to(device)
        labels = torch.stack([torch.from_numpy(lbl).long() for lbl in batch['lbl']], dim=0).to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics - Convert PyTorch tensors to NumPy arrays
        preds = outputs.argmax(dim=1)
        
        # Convert to CPU and then to numpy arrays for the metric
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        metric.update(labels_np, preds_np)
        scores = metric.get_scores()
        metric.reset()
        
        print(f"Iteration {iteration+1}/{max_iterations}, Loss: {loss.item():.4f}, IoU: {scores['iou_mean']:.4f}")
        
        # On the last iteration, store predictions for visualization
        if iteration == max_iterations - 1:
            final_preds = preds_np
            final_batch = batch
    
    # Visualize predictions
    visualize_batch(final_batch, final_preds)
    
    print("Training iterations completed successfully")

def main():
    """Main test function"""
    print("=== ISIC Dataset Pipeline Test ===")
    
    # Step 1: Test data loading
    train_loader = test_data_loading()
    
    # Step 2: Test model initialization
    model = test_model_initialization()
    
    # Step 3: Test training iterations
    test_training_iteration(model, train_loader)
    
    print("\n=== All Tests Passed Successfully ===")
    print("The ISIC dataset pipeline is working correctly!")

if __name__ == "__main__":
    main() 