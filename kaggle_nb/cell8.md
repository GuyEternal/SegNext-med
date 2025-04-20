# Cell 8: Load best model with CrossNeXt decoder and run inference on test set

import torch
import numpy as np
import matplotlib.pyplot as plt
import imgviz
from tqdm import tqdm

# Create data lists for testing
test_data = ISICDataset(
    test_paths[0], 
    test_paths[1], 
    config['img_height'], 
    config['img_width'],
    False,  # No augmentation for testing
    config['Normalize_data']
)

# Create test data loader
test_loader = DataLoader(
    test_data, 
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=config['num_workers'],
    collate_fn=collate, 
    pin_memory=config['pin_memory']
)

print(f"Test samples: {len(test_data)}")

# Load best model checkpoint - use exact same path as saved in cell7.md
checkpoint_path = f"{config['checkpoint_path']}{config['experiment_name']}.pth"
print(f"Looking for checkpoint at: {checkpoint_path}")

if os.path.exists(checkpoint_path):
    # IMPORTANT: Use the exact same architecture parameters as in cell6.md
    # This ensures compatibility with the saved checkpoint
    embed_dims = [32, 64, 160, 256]
    depths = [3, 2, 2, 2]  # Use [3, 2, 2, 2] from cell6.md, not the default [3, 3, 5, 2]
    dec_channels = 128  # Use 128 from cell6.md, not the default 256
    
    # Log the model architecture parameters being used
    print(f"Using model architecture parameters from training:")
    print(f"  embed_dims: {embed_dims}")
    print(f"  depths: {depths}")
    print(f"  dec_outChannels: {dec_channels}")
    print(f"  CrossNeXt decoder with {config.get('crossnext_num_heads', 8)} attention heads")
    
    # Initialize a new model instance with params from training
    best_model = SegNext(
        num_classes=config['num_classes'],
        in_channnels=config['input_channels'],
        embed_dims=embed_dims,
        ffn_ratios=[4, 4, 4, 4],
        depths=depths,
        num_stages=4,
        dec_outChannels=dec_channels,
        drop_path=float(config['stochastic_drop_path']),
        config=config  # Pass config to use CrossNeXt decoder parameters
    )
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint keys to debug
    print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
    
    # Handle DataParallel wrapper if needed
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        if isinstance(model_state, dict) and any(k.startswith('module.') for k in model_state.keys()):
            # Model was saved with DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            best_model.load_state_dict(new_state_dict)
        else:
            # Model was saved without DataParallel
            best_model.load_state_dict(model_state)
    elif 'state_dict' in checkpoint:  # Alternative key name
        best_model.load_state_dict(checkpoint['state_dict'])
    else:
        print("WARNING: No model state dictionary found in checkpoint!")
    
    best_model = best_model.to(device)
    
    # Extract training info using the exact keys used in cell7.md when saving
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    best_iou = checkpoint.get('iou', 'unknown')  # This is the key used in save_chkpt
    
    # Print checkpoint info
    print(f"Model loaded successfully from checkpoint")
    if epoch != 'unknown':
        print(f"Training epoch: {epoch}")
    if loss != 'unknown' and isinstance(loss, (int, float)):
        print(f"Training loss: {loss:.4f}")
    if best_iou != 'unknown' and isinstance(best_iou, (int, float)):
        print(f"Validation IoU: {best_iou:.4f}")
    
    # Set model to evaluation mode
    best_model.eval()
    
    # Initialize evaluation metric
    test_metric = ConfusionMatrix(config['num_classes'])
    
    # Run inference on test set
    test_ious = []
    test_visualizations = []
    
    with torch.no_grad():
        for data_batch in tqdm(test_loader, desc="Testing"):
            try:
                # Stack the batch images
                input_images = np.stack(data_batch['img'])
                
                # Debug the input shape
                if len(test_visualizations) == 0:
                    print(f"Raw input shape: {input_images.shape}")
                
                # Check if channels are last (NHWC format)
                if input_images.shape[-1] == 3:  # If last dimension is 3 (RGB channels)
                    # Permute to channels-first format (NCHW) as PyTorch expects
                    input_images = np.transpose(input_images, (0, 3, 1, 2))
                    if len(test_visualizations) == 0:
                        print(f"Converted to channels-first format: {input_images.shape}")
                
                # Move data to device AND convert to float32
                images = torch.from_numpy(input_images).float().to(device)
                labels = torch.from_numpy(np.stack(data_batch['lbl'])).to(device, dtype=torch.long)
                
                # Forward pass
                outputs = best_model(images)
                
                # Get predictions
                _, predictions = torch.max(outputs, 1)
                
                # Update metrics
                test_metric.update(labels.cpu().numpy(), predictions.cpu().numpy())
                
                # Save first batch visualizations
                if len(test_visualizations) < 3:
                    for i in range(min(images.size(0), 3 - len(test_visualizations))):
                        # Convert back to channels-last for visualization
                        img = images[i].cpu().numpy()
                        if img.shape[0] == 3:  # If channels-first
                            img = np.transpose(img, (1, 2, 0))  # Convert to channels-last
                        
                        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                        gt = labels[i].cpu().numpy()
                        pred = predictions[i].cpu().numpy()
                        test_visualizations.append((img, gt, pred))
                
                # Free memory after processing each batch
                del images, labels, outputs, predictions
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                continue
        
        if test_visualizations:
            # Get final test scores
            scores = test_metric.get_scores()
            print("Available metric keys:", list(scores.keys()))
            test_metric.reset()
            
            # Print test results safely checking for keys
            print("\nTest Results:")
            if 'iou_mean' in scores:
                print(f"Test IoU: {scores['iou_mean']:.4f}")
            if 'iou' in scores:
                print(f"Test IoU by class: {', '.join([f'{iou:.4f}' for iou in scores['iou']])}")
            # Safely check for pixel accuracy key (may be named differently)
            if 'pix_acc' in scores:
                print(f"Test Pixel Accuracy: {scores['pix_acc']:.4f}")
            elif 'pixel_acc' in scores:
                print(f"Test Pixel Accuracy: {scores['pixel_acc']:.4f}")
            elif 'acc' in scores:
                print(f"Test Accuracy: {scores['acc']:.4f}")
            else:
                # Try to use confusion matrix if available
                conf_matrix = scores.get('overall_confusion_matrix', None)
                if conf_matrix is not None and isinstance(conf_matrix, np.ndarray):
                    # Calculate accuracy from confusion matrix
                    correct_pixels = np.sum(conf_matrix.diagonal())
                    total_pixels = np.sum(conf_matrix)
                    if total_pixels > 0:
                        accuracy = correct_pixels / total_pixels
                        print(f"Test Pixel Accuracy (calculated): {accuracy:.4f}")
                    else:
                        print("Unable to calculate pixel accuracy (no pixels in confusion matrix)")
                else:
                    print("Unable to calculate pixel accuracy (no confusion matrix available)")
            
            # Create figure for visualizations
            if len(test_visualizations) > 0:
                # Improved visualization with color coding and overlay
                fig_rows = len(test_visualizations)
                fig, axes = plt.subplots(fig_rows, 3, figsize=(15, 5*fig_rows))
                
                # Handle case of single sample
                if fig_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, (img, gt, pred) in enumerate(test_visualizations):
                    # Original image
                    axes[i, 0].imshow(img)
                    axes[i, 0].set_title("Image")
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
                plt.savefig('test_predictions.png')
                plt.show()
                
                # Create overlay visualizations using imgviz
                for i, (img, gt, pred) in enumerate(test_visualizations):
                    # Create label visualization with imgviz
                    label_gt = np.zeros_like(gt, dtype=np.int32)
                    label_gt[gt == 1] = 1  # Convert binary mask to label
                    
                    label_pred = np.zeros_like(pred, dtype=np.int32)
                    label_pred[pred == 1] = 1  # Convert binary mask to label
                    
                    # Create overlay vizualizations
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
                    plt.savefig(f'overlay_visualization_{i+1}.png')
                    plt.show()
                
                # Print mask statistics
                for i, (_, gt, pred) in enumerate(test_visualizations):
                    gt_non_zero = np.count_nonzero(gt)
                    gt_percentage = (gt_non_zero / gt.size) * 100
                    pred_non_zero = np.count_nonzero(pred)
                    pred_percentage = (pred_non_zero / pred.size) * 100
                    print(f"Sample {i+1}:")
                    print(f"  Ground truth: {gt_non_zero} non-zero pixels ({gt_percentage:.2f}% of image)")
                    print(f"  Prediction: {pred_non_zero} non-zero pixels ({pred_percentage:.2f}% of image)")
            else:
                print("No test samples were visualized.")
else:
    print(f"Checkpoint file {checkpoint_path} not found. Training may not have saved a checkpoint yet.") 