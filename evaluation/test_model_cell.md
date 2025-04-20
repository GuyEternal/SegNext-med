# Cell: Test SegNext model with pre-trained checkpoint

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import imgviz
from tqdm import tqdm
import gc

# Create output directory for test results
output_dir = 'test_results'
os.makedirs(output_dir, exist_ok=True)

# Path to the checkpoint file
checkpoint_path = '/kaggle/input/checkpoint_till_train/isic_segmentation_model_after_train_benchmark_on_isic2017_segnextbasemodel.pth'

# Check if the checkpoint exists
print(f"Checking for checkpoint at: {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found at {checkpoint_path}")
    print("Please make sure the checkpoint file is available at the specified path")
    # Exit the cell gracefully
    raise ValueError("Checkpoint file not found")
else:
    print(f"✅ Checkpoint found at {checkpoint_path}")

# Set the working directory
%cd /kaggle/working/SegNext-med

# Load the Kaggle configuration
with open('config_kaggle.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# Set CUDA environment variables for multi-GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use']

# Check for GPU availability and display GPU information
print(f"CUDA available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Number of available GPUs: {device_count}")

for i in range(device_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Set the device
if device_count > 1:
    print("Using multiple GPUs for testing")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using single device: {device}")

# Import required modules from the repository
from dataloader import GEN_DATA_LISTS, ISICDataset
from data_utils import collate, pallet_isic
from model import SegNext
from metrics import ConfusionMatrix
from gray2color import gray2color

# Setup visualization functions
g2c = lambda x : gray2color(x, use_pallet='custom', custom_pallet=pallet_isic)

# Generate data lists from the dataset
data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
_, _, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()

print("Dataset statistics:")
data_lists.get_filecounts()

# Create test dataset
test_data = ISICDataset(
    test_paths[0], 
    test_paths[1], 
    config['img_height'], 
    config['img_width'],
    False,  # No augmentation for testing
    config['Normalize_data']
)

# Create test data loader with smaller batch size for testing
batch_size = min(config['batch_size'], 4)  # Smaller batch size for testing
test_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=config['num_workers'],
    collate_fn=collate, 
    pin_memory=config['pin_memory']
)

print(f"Test samples: {len(test_data)}")
print(f"Loading checkpoint from: {checkpoint_path}")

# Get model architecture parameters from config
embed_dims = config.get('embed_dims', [32, 64, 160, 256])
depths = config.get('depths', [3, 3, 5, 2])
dec_channels = config.get('dec_outChannels', 256)

print(f"Using model architecture:")
print(f"  embed_dims: {embed_dims}")
print(f"  depths: {depths}")
print(f"  dec_outChannels: {dec_channels}")

# Initialize model
model = SegNext(
    num_classes=config['num_classes'],
    in_channnels=config['input_channels'],
    embed_dims=embed_dims,
    ffn_ratios=[4, 4, 4, 4],
    depths=depths,
    num_stages=4,
    dec_outChannels=dec_channels,
    drop_path=float(config['stochastic_drop_path']),
    config=config
)

# Load checkpoint
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        print("WARNING: Using entire checkpoint as model state dictionary")
        model_state = checkpoint
    
    # Handle DataParallel wrapped models
    if isinstance(model_state, dict) and any(k.startswith('module.') for k in model_state.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded model from DataParallel checkpoint")
    else:
        # Try direct loading
        try:
            model.load_state_dict(model_state)
            print("Loaded model successfully")
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(model_state, strict=False)
            print("Loaded model with strict=False (some weights may not be loaded)")
    
    # Extract training info if available
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    best_iou = checkpoint.get('iou', 'unknown')
    
    if epoch != 'unknown':
        print(f"Training epoch: {epoch}")
    if loss != 'unknown' and isinstance(loss, (int, float)):
        print(f"Training loss: {loss:.4f}")
    if best_iou != 'unknown' and isinstance(best_iou, (int, float)):
        print(f"Validation IoU: {best_iou:.4f}")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    # Raise the error to stop cell execution
    raise

# Move model to device
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Initialize evaluation metric
test_metric = ConfusionMatrix(config['num_classes'])

# Initialize lists for visualization
test_visualizations = []

# Run inference
with torch.no_grad():
    for batch_idx, data_batch in enumerate(tqdm(test_loader, desc="Testing")):
        try:
            # Stack the batch images
            input_images = np.stack(data_batch['img'])
            
            # Debug input format
            if batch_idx == 0:
                print(f"Raw input shape: {input_images.shape}")
            
            # Check input image format and convert if needed
            if input_images.shape[-1] == 3:  # If last dimension is 3 (RGB channels)
                # Permute to channels-first format (NCHW) as PyTorch expects
                input_images = np.transpose(input_images, (0, 3, 1, 2))
                if batch_idx == 0:
                    print(f"Converted to channels-first format: {input_images.shape}")
            
            # Move data to device
            images = torch.from_numpy(input_images).float().to(device)
            labels = torch.from_numpy(np.stack(data_batch['lbl'])).to(device, dtype=torch.long)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Update metrics
            test_metric.update(labels.cpu().numpy(), predictions.cpu().numpy())
            
            # Save visualizations for the first few batches
            if len(test_visualizations) < 5:  # Save up to 5 samples
                for i in range(min(images.size(0), 5 - len(test_visualizations))):
                    # Convert image back to channels-last for visualization
                    img = images[i].cpu().numpy()
                    if img.shape[0] == 3:  # If channels-first
                        img = np.transpose(img, (1, 2, 0))  # Convert to channels-last
                    
                    # Scale image for visualization
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                    gt = labels[i].cpu().numpy()
                    pred = predictions[i].cpu().numpy()
                    
                    # Store for visualization
                    test_visualizations.append((img, gt, pred))
            
            # Free up memory
            del images, labels, outputs, predictions
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Continue to next batch
            continue

# Get final test scores
scores = test_metric.get_scores()
print("\nTest Results:")
print(f"Available metrics: {list(scores.keys())}")

# Print test results
if 'iou_mean' in scores:
    print(f"Test IoU: {scores['iou_mean']:.4f}")
if 'iou' in scores:
    print(f"Test IoU by class: {', '.join([f'{iou:.4f}' for iou in scores['iou']])}")

# Print pixel accuracy
for acc_key in ['pix_acc', 'pixel_acc', 'acc']:
    if acc_key in scores:
        print(f"Test Pixel Accuracy: {scores[acc_key]:.4f}")
        break

# Create visualizations
if test_visualizations:
    # Create figure for visualizations
    fig_rows = len(test_visualizations)
    fig, axes = plt.subplots(fig_rows, 3, figsize=(15, 5*fig_rows))
    
    # Handle case of single sample
    if fig_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, gt, pred) in enumerate(test_visualizations):
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')
        
        # Create colored masks for better visibility
        colored_gt = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        colored_gt[gt == 1] = [255, 0, 0]  # Red for ground truth
        
        colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        colored_pred[pred == 1] = [0, 255, 0]  # Green for prediction
        
        # Ground truth - colored
        axes[i, 1].imshow(colored_gt)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction - colored
        axes[i, 2].imshow(colored_pred)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
    plt.show()
    
    # Also create overlay visualizations using imgviz
    for i, (img, gt, pred) in enumerate(test_visualizations):
        # Create label visualization with imgviz
        label_gt = np.zeros_like(gt, dtype=np.int32)
        label_gt[gt == 1] = 1  # Convert binary mask to label
        
        label_pred = np.zeros_like(pred, dtype=np.int32)
        label_pred[pred == 1] = 1  # Convert binary mask to label
        
        # Visualization with overlay
        viz_gt = imgviz.label2rgb(label_gt, img, alpha=0.5, label_names={0: "Background", 1: "Lesion"})
        viz_pred = imgviz.label2rgb(label_pred, img, alpha=0.5, label_names={0: "Background", 1: "Lesion"})
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(viz_gt)
        plt.title("Ground Truth Overlay")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(viz_pred)
        plt.title("Prediction Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overlay_visualization_{i+1}.png'))
        plt.show()
        
        # Print mask statistics
        gt_non_zero = np.count_nonzero(gt)
        gt_percentage = (gt_non_zero / gt.size) * 100
        pred_non_zero = np.count_nonzero(pred)
        pred_percentage = (pred_non_zero / pred.size) * 100
        print(f"Sample {i+1}:")
        print(f"  Ground truth: {gt_non_zero} non-zero pixels ({gt_percentage:.2f}% of image)")
        print(f"  Prediction: {pred_non_zero} non-zero pixels ({pred_percentage:.2f}% of image)")

# Save test results to file
with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
    f.write(f"Test IoU: {scores.get('iou_mean', 'N/A')}\n")
    if 'iou' in scores:
        f.write(f"Test IoU by class: {scores['iou']}\n")
    for acc_key in ['pix_acc', 'pixel_acc', 'acc']:
        if acc_key in scores:
            f.write(f"Test Pixel Accuracy: {scores[acc_key]}\n")
            break

print(f"\nTest completed! Results saved to {output_dir}/test_results.txt") 