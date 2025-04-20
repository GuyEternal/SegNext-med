# Cell 7: Train the SegNext model on ISIC dataset

import time
import gc
import psutil
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import imgviz

# Function to get system memory usage
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024**2  # MB

# Function to get GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
            gpu_memory.append((memory_allocated, memory_reserved))
        return gpu_memory
    return []

# Create directory for saving visualizations (only one file will be saved)
os.makedirs('visualizations', exist_ok=True)

# Initialize tracking variables
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou = []
losses, train_ious, val_ious = [], [], []
# Store only the last few epochs' memory usage to maintain constant memory
gpu_memory_usage = []
MAX_MEMORY_RECORDS = 10  # Only keep records for the last 10 epochs
start_time = time.time()

# Log initial resource usage
initial_ram = get_memory_usage()
initial_gpu = get_gpu_memory()
print(f"Initial RAM usage: {initial_ram:.2f} MB")
for i, (allocated, reserved) in enumerate(initial_gpu):
    print(f"Initial GPU {i} memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

# Save checkpoint function with memory optimization
def save_checkpoint(model, optimizer, epoch, loss, iou):
    # Clear CUDA cache before saving to reduce memory pressure
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save checkpoint (overwrites the existing file)
    mu.save_chkpt(model, optimizer, epoch, loss, iou)
    
    # Force garbage collection
    gc.collect()
    
    print(f'Checkpoint saved with IoU: {iou:.4f}')
    print(f'Current RAM usage: {get_memory_usage():.2f} MB')
    for i, (allocated, reserved) in enumerate(get_gpu_memory()):
        print(f"GPU {i} memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

# Training loop
for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    # TRAINING PHASE
    model.train()
    train_loss = []
    train_iou = []
    
    # Progress bar for training
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
    
    # Log memory usage at the start of the epoch
    if epoch == 0:
        print(f"Memory usage before first batch: {get_memory_usage():.2f} MB")
        for i, (allocated, reserved) in enumerate(get_gpu_memory()):
            print(f"GPU {i} memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    
    for step, data_batch in enumerate(pbar):
        # Update learning rate
        scheduler(optimizer, step, epoch)
        
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
            f'Epoch {epoch+1}/{config["epochs"]} - Loss: {loss_value:.4f} - IoU: {iou["iou_mean"]:.4f}'
        )
        
        # Log memory usage for the first batch of the first epoch
        if epoch == 0 and step == 0:
            print(f"Memory usage after first batch: {get_memory_usage():.2f} MB")
            for i, (allocated, reserved) in enumerate(get_gpu_memory()):
                print(f"GPU {i} memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
        
        # Free up memory periodically
        if step % 10 == 0 and step > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Average training metrics for this epoch
    avg_train_loss = np.nanmean(train_loss)
    avg_train_iou = np.nanmean(train_iou)
    losses.append(avg_train_loss)
    train_ious.append(avg_train_iou)
    
    # Maintain fixed-size lists for memory efficiency (only when they grow too large)
    if len(losses) > MAX_MEMORY_RECORDS:
        losses = losses[-MAX_MEMORY_RECORDS:]
        train_ious = train_ious[-MAX_MEMORY_RECORDS:]
        val_ious = val_ious[-MAX_MEMORY_RECORDS:]
    
    # VALIDATION PHASE (every epoch)
    model.eval()
    val_iou = []
    
    # Progress bar for validation
    vbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]')
    
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
                f'Epoch {epoch+1}/{config["epochs"]} - Val IoU: {viou["iou_mean"]:.4f}'
            )
    
    # Average validation metrics for this epoch
    avg_val_iou = np.nanmean(val_iou)
    val_ious.append(avg_val_iou)
    total_avg_viou.append(avg_val_iou)
    # Keep only the best validation IoU value to save memory
    if len(total_avg_viou) > MAX_MEMORY_RECORDS:
        curr_viou = np.nanmax(total_avg_viou)
        # Replace the list with just the max value to save memory
        total_avg_viou = [curr_viou]
    else:
        curr_viou = np.nanmax(total_avg_viou)
    
    # Get sample prediction for visualization
    img, gt, pred = evaluator.get_sample_prediction()
    
    # Improved visualization with colored masks - save only one file that gets overwritten
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
    
    # Save the visualization (single file that gets overwritten)
    plt.tight_layout()
    plt.savefig('visualizations/latest_prediction.png')
    plt.close()
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Track GPU memory usage (limited to a fixed number of entries)
    current_gpu_memory = [(allocated, reserved) for allocated, reserved in get_gpu_memory()]
    gpu_memory_usage.append(current_gpu_memory)
    if len(gpu_memory_usage) > MAX_MEMORY_RECORDS:
        gpu_memory_usage = gpu_memory_usage[-MAX_MEMORY_RECORDS:]
    
    # Print epoch summary
    print(f'\nEpoch {epoch+1} Summary:')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Training IoU: {avg_train_iou:.4f}')
    print(f'Validation IoU: {avg_val_iou:.4f}')
    print(f'Best Validation IoU: {curr_viou:.4f}')
    print(f'Epoch time: {epoch_time:.2f}s')
    
    # Print resource usage
    ram_usage = get_memory_usage()
    print(f"RAM usage: {ram_usage:.2f} MB")
    for i, (allocated, reserved) in enumerate(current_gpu_memory):
        print(f"GPU {i} memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    
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
        save_checkpoint(model, optimizer, epoch, avg_train_loss, best_iou)
    
    # Free memory before plotting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Plot training progress
    clear_output(wait=True)
    
    plt.figure(figsize=(20, 10))
    # Training metrics
    plt.subplot(2, 2, 1)
    plt.plot(range(epoch+2-len(losses), epoch+2), losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(epoch+2-len(train_ious), epoch+2), train_ious, 'r-', label='Training IoU')
    plt.plot(range(epoch+2-len(val_ious), epoch+2), val_ious, 'g-', label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Metrics')
    plt.legend()
    plt.grid(True)
    
    # Resource usage
    if gpu_memory_usage:
        plt.subplot(2, 2, 3)
        for i in range(torch.cuda.device_count()):
            gpu_mem = [mem[i][0] for mem in gpu_memory_usage]  # Allocated memory
            plt.plot(range(epoch+2-len(gpu_mem), epoch+2), gpu_mem, label=f'GPU {i} Allocated')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage')
        plt.legend()
        plt.grid(True)
    
    # Computation time
    plt.subplot(2, 2, 4)
    elapsed_time = time.time() - start_time
    time_per_epoch = elapsed_time / (epoch + 1)
    estimated_total = time_per_epoch * config['epochs']
    remaining_time = estimated_total - elapsed_time
    
    labels = ['Elapsed', 'Remaining', 'Estimated Total']
    times = [elapsed_time/60, remaining_time/60, estimated_total/60]  # Convert to minutes
    plt.bar(labels, times, color=['blue', 'orange', 'green'])
    plt.ylabel('Time (minutes)')
    plt.title('Training Time')
    
    plt.tight_layout()
    # Save the plot (single file that gets overwritten)
    plt.savefig('training_progress.png')
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

# Print training summary
total_time = time.time() - start_time
print(f'\nTraining completed in {total_time/60:.2f} minutes')
print(f'Best validation IoU: {best_iou:.4f}')
print(f'Final training loss: {losses[-1]:.4f}')
print(f'Checkpoint saved at: {config["checkpoint_path"]}{config["experiment_name"]}.pth')

# Final resource usage
final_ram = get_memory_usage()
final_gpu = get_gpu_memory()
print(f"Final RAM usage: {final_ram:.2f} MB (Change: {final_ram - initial_ram:.2f} MB)")
for i, ((final_allocated, final_reserved), (initial_allocated, initial_reserved)) in enumerate(zip(final_gpu, initial_gpu)):
    print(f"Final GPU {i} memory: {final_allocated:.2f} MB allocated (Change: {final_allocated - initial_allocated:.2f} MB)")

# Do not save GPU memory usage data to keep memory constant
# Instead, print a summary of peak memory usage
if gpu_memory_usage:
    print("\nPeak GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        peak_mem = max([mem[i][0] for mem in gpu_memory_usage])
        print(f"GPU {i} peak allocated: {peak_mem:.2f} MB") 