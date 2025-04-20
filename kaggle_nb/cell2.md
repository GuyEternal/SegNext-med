# Cell 2: Create necessary directories and modify config for Kaggle

import os
import yaml
import shutil

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load and modify config for Kaggle environment
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Update config for Kaggle
config['data_dir'] = "/kaggle/input/processed-dataset/processed_isic_dataset"
config['log_directory'] = "/kaggle/working/SegNext-med/logs/"
config['checkpoint_path'] = "/kaggle/working/SegNext-med/checkpoints/"
config['batch_size'] = 2  # Reduce batch size for Kaggle free tier
config['num_workers'] = 2  # Adjust workers for Kaggle
config['epochs'] = 40  # Reduce epochs for demo
config['gpus_to_use'] = '0, 1'

# Add CrossNext decoder parameters
# CrossNext-T (Tiny) variant parameters from the paper
config['embed_dims'] = [32, 64, 160, 256]    # Encoder channels for Tiny variant
config['depths'] = [3, 3, 5, 2]              # Encoder blocks for Tiny variant
config['decoder_channels'] = 64              # Decoder channels for Tiny variant

# CrossNext attention parameters
config['crossnext_num_heads'] = 8            # Number of attention heads
config['crossnext_kernel_sizes'] = [7, 11, 21]  # Kernel sizes for multi-scale strip convolutions
config['crossnext_norm_type'] = 'WithBias'   # Type of normalization in attention

# Training hyperparameters from paper
config['learning_rate'] = 0.0002             # Initial learning rate
config['WEIGHT_DECAY'] = 1e-4                # Weight decay

# Save the updated config
with open('config_kaggle.yaml', 'w') as file:
    yaml.dump(config, file)

print("Configuration updated for Kaggle environment.")
print(f"Using dataset from: {config['data_dir']}")
print(f"Logs will be saved to: {config['log_directory']}")
print(f"Checkpoints will be saved to: {config['checkpoint_path']}")
print("\nCrossNext configuration:")
print(f"- Encoder dimensions: {config['embed_dims']}")
print(f"- Encoder depths: {config['depths']}")
print(f"- Decoder channels: {config['decoder_channels']}")
print(f"- Number of attention heads: {config['crossnext_num_heads']}")
print(f"- Kernel sizes: {config['crossnext_kernel_sizes']}")
print(f"- Normalization type: {config['crossnext_norm_type']}")
print(f"- Learning rate: {config['learning_rate']}")
print(f"- Weight decay: {config['WEIGHT_DECAY']}") 