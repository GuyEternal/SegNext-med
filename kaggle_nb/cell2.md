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

# Save the updated config
with open('config_kaggle.yaml', 'w') as file:
    yaml.dump(config, file)

print("Configuration updated for Kaggle environment.")
print(f"Using dataset from: {config['data_dir']}")
print(f"Logs will be saved to: {config['log_directory']}")
print(f"Checkpoints will be saved to: {config['checkpoint_path']}") 