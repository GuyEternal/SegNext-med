# Cell 5: Create data loaders for training and validation

# Create training dataset
train_data = ISICDataset(
    train_paths[0], 
    train_paths[1], 
    config['img_height'], 
    config['img_width'],
    config['Augment_data'], 
    config['Normalize_data']
)

# Configure DataLoader parameters
dataloader_kwargs = {
    'batch_size': config['batch_size'],
    'shuffle': config['Shuffle_data'],
    'num_workers': config['num_workers'],
    'collate_fn': collate,
    'pin_memory': config['pin_memory']
}

# Add prefetch_factor and persistent_workers if using workers
if config['num_workers'] > 0:
    dataloader_kwargs.update({
        'prefetch_factor': 2,
        'persistent_workers': True
    })

# Create training data loader
train_loader = DataLoader(train_data, **dataloader_kwargs)

# Create validation dataset
val_data = ISICDataset(
    val_paths[0], 
    val_paths[1], 
    config['img_height'], 
    config['img_width'],
    False,  # No augmentation for validation
    config['Normalize_data']
)

# Set shuffle to False for validation
val_dataloader_kwargs = dataloader_kwargs.copy()
val_dataloader_kwargs['shuffle'] = False

# Create validation data loader
val_loader = DataLoader(val_data, **val_dataloader_kwargs)

# Print dataset information
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Batch size: {config['batch_size']}")
print(f"Steps per epoch: {len(train_loader)}")

# Visualize a batch of samples with improved visualization
try:
    batch = next(iter(train_loader))
    s = 255
    
    # Get number of samples to display (up to batch size)
    batch_size = min(config['batch_size'], len(batch['img']))
    
    # Create a figure with rows for each sample
    plt.figure(figsize=(15, 5 * batch_size))
    
    for i in range(batch_size):
        # Get image and mask for this sample
        img = (batch['img'][i] * s).astype(np.uint8)
        mask = batch['lbl'][i]
        
        # Create colored mask for better visibility
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 1] = [255, 0, 0]  # Red for lesion area
        
        # Plot original image
        plt.subplot(batch_size, 3, i*3 + 1)
        plt.imshow(img)
        plt.title(f'Sample {i+1}: Image')
        plt.axis('off')
        
        # Plot grayscale mask
        plt.subplot(batch_size, 3, i*3 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Sample {i+1}: Mask (Grayscale)')
        plt.axis('off')
        
        # Plot colored mask
        plt.subplot(batch_size, 3, i*3 + 3)
        plt.imshow(colored_mask)
        plt.title(f'Sample {i+1}: Mask (Colored)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch.png')
    plt.show()
    
    # Print mask statistics for verification
    for i in range(batch_size):
        mask = batch['lbl'][i]
        non_zero = np.count_nonzero(mask)
        percentage = (non_zero / mask.size) * 100
        print(f"Sample {i+1} mask stats: {non_zero} non-zero pixels ({percentage:.2f}% of image)")
    
    print("Sample batch visualization saved as 'sample_batch.png'")
except Exception as e:
    print(f"Error visualizing batch: {e}") 