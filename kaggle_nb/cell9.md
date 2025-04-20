# Cell 9: Resume training from a checkpoint with CrossNext decoder

import time
import yaml
import torch
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the checkpoint path to load from
checkpoint_path = f"{config['checkpoint_path']}{config['experiment_name']}.pth"
# Or manually specify a different checkpoint if needed:
# checkpoint_path = "/kaggle/working/SegNext-med/checkpoints/specific_checkpoint.pth"

# Check if checkpoint exists
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint file not found: {checkpoint_path}")
else:
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with same architecture from config
    resume_model = SegNext(
        num_classes=config['num_classes'],
        in_channnels=config['input_channels'],
        embed_dims=config['embed_dims'],         # [32, 64, 160, 256] for Tiny variant
        ffn_ratios=[4, 4, 4, 4],
        depths=config['depths'],                 # [3, 3, 5, 2] for Tiny variant
        num_stages=4,
        dec_outChannels=config['decoder_channels'],  # 64 for Tiny variant
        drop_path=float(config['stochastic_drop_path']),
        config=config  # Pass config to use CrossNext decoder parameters
    )
    
    # Print model architecture for verification
    print(f"Resuming with model architecture:")
    print(f"  embed_dims: {config['embed_dims']}")
    print(f"  depths: {config['depths']}")
    print(f"  dec_outChannels: {config['decoder_channels']}")
    print(f"  CrossNext decoder with {config['crossnext_num_heads']} attention heads")
    
    # Handle DataParallel wrapper if needed
    if isinstance(checkpoint['model_state_dict'], dict) and list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
        # Model was saved with DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        resume_model.load_state_dict(new_state_dict)
    else:
        # Model was saved without DataParallel
        resume_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    resume_model = resume_model.to(device)
    
    # Initialize optimizer
    resume_optimizer = torch.optim.AdamW(
        [{'params': resume_model.parameters(), 'lr': config['learning_rate']}], 
        weight_decay=config['WEIGHT_DECAY']
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        resume_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get previous training info
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_iou = checkpoint['best_iou']
    
    print(f"Resuming from epoch {start_epoch} with best IoU: {best_iou:.4f}")
    
    # Initialize metrics and training utilities
    metric = ConfusionMatrix(config['num_classes'])
    mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
    
    # Initialize scheduler
    scheduler = LR_Scheduler(
        config['lr_schedule'],
        config['learning_rate'],
        config['epochs'],
        iters_per_epoch=len(train_loader),
        warmup_epochs=config['warmup_epochs']
    )
    
    # Initialize trainer and evaluator
    trainer = Trainer(resume_model, config['batch_size'], resume_optimizer, criterion, metric)
    evaluator = Evaluator(resume_model, metric)
    
    # Set up tracking variables
    # Load previous metrics if available (if you saved them)
    try:
        # Try to load previous metrics if you saved them
        metrics_path = f"{config['checkpoint_path']}metrics_{config['experiment_name']}.pth"
        if os.path.exists(metrics_path):
            metrics_data = torch.load(metrics_path)
            losses = metrics_data.get('losses', [])
            train_ious = metrics_data.get('train_ious', [])
            val_ious = metrics_data.get('val_ious', [])
            total_avg_viou = metrics_data.get('total_avg_viou', [])
            curr_viou = best_iou
            print(f"Loaded previous metrics with {len(losses)} epochs of history")
        else:
            # Initialize new metrics
            losses, train_ious, val_ious = [], [], []
            total_avg_viou = []
            curr_viou = best_iou
            print("No previous metrics found, initializing new tracking variables")
    except Exception as e:
        print(f"Error loading metrics: {e}")
        # Initialize new metrics
        losses, train_ious, val_ious = [], [], []
        total_avg_viou = []
        curr_viou = best_iou
    
    # Set number of additional epochs to train
    additional_epochs = 10  # Change this to how many more epochs you want to train
    total_epochs = start_epoch + additional_epochs
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Training loop for additional epochs
    start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        
        # TRAINING PHASE
        resume_model.train()
        train_loss = []
        train_iou = []
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
        
        for step, data_batch in enumerate(pbar):
            # Update learning rate
            scheduler(resume_optimizer, step, epoch)
            
            # Perform training step
            loss_value = trainer.training_step(data_batch)
            
            # Get training metrics
            iou = trainer.get_scores()
            trainer.reset_metric()
            
            # Track metrics
            train_loss.append(loss_value)
            train_iou.append(iou['iou_mean'])
            
            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch+1}/{total_epochs} - Loss: {loss_value:.4f} - IoU: {iou["iou_mean"]:.4f}'
            )
        
        # Average training metrics for this epoch
        avg_train_loss = np.nanmean(train_loss)
        avg_train_iou = np.nanmean(train_iou)
        losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        
        # VALIDATION PHASE
        resume_model.eval()
        val_iou = []
        
        # Progress bar for validation
        vbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
        
        for step, val_batch in enumerate(vbar):
            with torch.no_grad():
                # Perform validation step
                evaluator.eval_step(val_batch)
                viou = evaluator.get_scores()
                evaluator.reset_metric()
                
                # Track metrics
                val_iou.append(viou['iou_mean'])
                
                # Update progress bar
                vbar.set_description(
                    f'Epoch {epoch+1}/{total_epochs} - Val IoU: {viou["iou_mean"]:.4f}'
                )
        
        # Average validation metrics for this epoch
        avg_val_iou = np.nanmean(val_iou)
        val_ious.append(avg_val_iou)
        total_avg_viou.append(avg_val_iou)
        curr_viou = np.nanmax(total_avg_viou)
        
        # Get sample prediction for visualization
        img, gt, pred = evaluator.get_sample_prediction()
        
        # Improved visualization with colored masks
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')
        
        # Ground truth - grayscale
        plt.subplot(2, 3, 2)
        plt.title('Ground Truth (Grayscale)')
        plt.imshow(gt, cmap='gray')
        plt.axis('off')
        
        # Prediction - grayscale
        plt.subplot(2, 3, 3)
        plt.title('Prediction (Grayscale)')
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        
        # Create colored versions of the masks
        colored_gt = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        colored_gt[gt == 1] = [255, 0, 0]  # Red for ground truth lesion
        
        colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        colored_pred[pred == 1] = [0, 255, 0]  # Green for predicted lesion
        
        # Leave the original image in second row first column
        plt.subplot(2, 3, 4)
        plt.title('Original Image (Repeated)')
        plt.imshow(img)
        plt.axis('off')
        
        # Ground truth - colored
        plt.subplot(2, 3, 5)
        plt.title('Ground Truth (Colored)')
        plt.imshow(colored_gt)
        plt.axis('off')
        
        # Prediction - colored
        plt.subplot(2, 3, 6)
        plt.title('Prediction (Colored)')
        plt.imshow(colored_pred)
        plt.axis('off')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(f'visualizations/resumed_prediction_epoch_{epoch+1}.png')
        plt.close()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Training IoU: {avg_train_iou:.4f}')
        print(f'Validation IoU: {avg_val_iou:.4f}')
        print(f'Best Validation IoU: {curr_viou:.4f}')
        print(f'Epoch time: {epoch_time:.2f}s')
        
        # Print mask statistics for verification
        gt_non_zero = np.count_nonzero(gt)
        gt_percentage = (gt_non_zero / gt.size) * 100
        pred_non_zero = np.count_nonzero(pred)
        pred_percentage = (pred_non_zero / pred.size) * 100
        print(f"Ground truth: {gt_non_zero} non-zero pixels ({gt_percentage:.2f}% of image)")
        print(f"Prediction: {pred_non_zero} non-zero pixels ({pred_percentage:.2f}% of image)")
        
        # Save checkpoint if model improved
        if curr_viou > best_iou:
            best_iou = curr_viou
            mu.save_chkpt(resume_model, resume_optimizer, epoch, avg_train_loss, best_iou)
            print(f'Checkpoint saved with IoU: {best_iou:.4f}')
        
        # Plot training progress
        clear_output(wait=True)
        
        # Plot with full history (including previous training)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(losses)+1), losses, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_ious)+1), train_ious, 'r-', label='Training IoU')
        plt.plot(range(1, len(val_ious)+1), val_ious, 'g-', label='Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('resumed_training_progress.png')
        plt.show()
        
        # Display the latest prediction with improved visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')
        
        # Ground truth - grayscale
        plt.subplot(2, 3, 2)
        plt.title('Ground Truth (Grayscale)')
        plt.imshow(gt, cmap='gray')
        plt.axis('off')
        
        # Prediction - grayscale
        plt.subplot(2, 3, 3)
        plt.title('Prediction (Grayscale)')
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        
        # Leave the original image in second row first column
        plt.subplot(2, 3, 4)
        plt.title('Original Image (Repeated)')
        plt.imshow(img)
        plt.axis('off')
        
        # Ground truth - colored
        plt.subplot(2, 3, 5)
        plt.title('Ground Truth (Colored)')
        plt.imshow(colored_gt)
        plt.axis('off')
        
        # Prediction - colored
        plt.subplot(2, 3, 6)
        plt.title('Prediction (Colored)')
        plt.imshow(colored_pred)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Save final metrics for future resuming
    metrics_data = {
        'losses': losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'total_avg_viou': total_avg_viou
    }
    metrics_path = f"{config['checkpoint_path']}metrics_{config['experiment_name']}.pth"
    torch.save(metrics_data, metrics_path)
    
    # Print training summary
    total_time = time.time() - start_time
    print(f'\nResumed training completed in {total_time/60:.2f} minutes')
    print(f'Best validation IoU: {best_iou:.4f}')
    print(f'Final training loss: {losses[-1]:.4f}')
    print(f'Checkpoint saved at: {config["checkpoint_path"]}{config["experiment_name"]}.pth')
    print(f'Metrics saved at: {metrics_path}') 