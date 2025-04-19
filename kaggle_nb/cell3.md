# Create updated config.yaml for Kaggle environment
%%writefile config.yaml
# Configuration file for training SegNeXt model on ISIC dataset in Kaggle

gpus_to_use: '0' # Use the available Kaggle GPU
DPI: 300
LOG_WANDB: False

project_name: 'SegNeXt'
experiment_name: 'isic_segmentation_model'

log_directory: "./logs/"
checkpoint_path: "./checkpoints/"

# Data loader parameters
data_dir: "/kaggle/input/small_isic_dataset/"  # Path to Kaggle dataset
# First is images, second is masks
sub_directories: ['images/', 'masks/']
Normalize_data: True
Shuffle_data: True
Augment_data: True
pin_memory: True
num_workers: 0  # Keep at 0 to avoid multiprocessing issues
num_classes: 2  # ISIC binary segmentation (background and lesion)
img_height: 512  # Standardized size for ISIC images
img_width: 512   # Standardized size for ISIC images
output_stride: 4
input_channels: 3
label_smoothing: 0.0
batch_size: 4  # Adjust if needed based on Kaggle GPU memory
WEIGHT_DECAY: 0.00005
stochastic_drop_path: 2e-1
layer_scaling_val: 1e-1

# learning rate
learning_rate: 0.001
lr_schedule: 'cos'
epochs: 20      # Reduced for Kaggle's time limits
warmup_epochs: 2
norm_typ: 'batch_norm'
BN_MOM: 0.9
SyncBN_MOM: 3e-4

# Hamburger Parameters
ham_channels: 512
put_cheese: True

DUAL: False
SPATIAL: TRUE
RAND_INIT: True

MD_S: 1
MD_D: 512
MD_R: 64

TRAIN_STEPS: 6
EVAL_STEPS: 6

INV_T: 1
BETA: 0.1
Eta: 0.9 