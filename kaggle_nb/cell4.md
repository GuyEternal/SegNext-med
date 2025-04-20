# Cell 4: Import necessary modules and prepare for training

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300

# Set the working directory
%cd /kaggle/working/SegNext-med

# Load the Kaggle configuration
with open('config_kaggle.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# Set CUDA environment variables for multi-GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # This should be '0,1' for dual GPUs
print(config['gpus_to_use'])

# Check for GPU availability and display GPU information
print(f"CUDA available: {torch.cuda.is_available()}")
[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
device_count = torch.cuda.device_count()
print(f"Number of available GPUs: {device_count}")
for i in range(device_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Set the device
if device_count > 1:
    print("Using multiple GPUs for training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if we're using synchronized batch norm for multi-GPU training
    if config.get('norm_typ') == 'sync_bn':
        print("Using synchronized batch normalization for multi-GPU training")
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using single device: {device}")

# Import required modules from the repository
from dataloader import GEN_DATA_LISTS, ISICDataset
from data_utils import collate, pallet_isic
from model import SegNext
from losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer, Evaluator, ModelUtils
from gray2color import gray2color

# Setup visualization functions
g2c = lambda x : gray2color(x, use_pallet='custom', custom_pallet=pallet_isic)

# Generate data lists from the dataset
data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
train_paths, val_paths, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()
data_lists.get_filecounts()

# Print summary of multi-GPU setup
print("\nTraining configuration summary:")
print(f"GPUs being used: {config['gpus_to_use']}")
print(f"Batch size (total): {config['batch_size']}")
if device_count > 1:
    print(f"Effective batch size per GPU: {config['batch_size'] // device_count}")
print(f"Model architecture: embed_dims={config.get('embed_dims')}")
print(f"Model architecture: depths={config.get('depths')}")
print(f"Normalization type: {config.get('norm_typ', 'batch_norm')}")
print(f"Number of workers: {config['num_workers']}")
print(f"Training epochs: {config['epochs']}") 