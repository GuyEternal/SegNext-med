```python
# Evaluate model on test set
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from torch.utils.data import DataLoader
mpl.rcParams['figure.dpi'] = 300

# Load configuration
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

from dataloader import GEN_DATA_LISTS, ISICDataset
from data_utils import collate, pallet_isic
from model import SegNext
from metrics import ConfusionMatrix
from utils import Evaluator, ModelUtils
from gray2color import gray2color

# Use ISIC dataset palette for visualization
palette = pallet_isic
g2c = lambda x: gray2color(x, use_pallet='custom', custom_pallet=palette)

# Get data paths
data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
_, _, test_paths = data_lists.get_splits()

# Create test dataset and dataloader
test_data = ISICDataset(
    test_paths[0], test_paths[1], 
    config['img_height'], config['img_width'],
    False, config['Normalize_data']
)

# Configure DataLoader
dataloader_kwargs = {
    'batch_size': config['batch_size'],
    'shuffle': False,
    'num_workers': config['num_workers'],
    'collate_fn': collate,
    'pin_memory': config['pin_memory']
}

test_loader = DataLoader(test_data, **dataloader_kwargs)

# Initialize the model
model = SegNext(
    num_classes=config['num_classes'], 
    in_channnels=3, 
    embed_dims=[32, 64, 460, 256],
    ffn_ratios=[4, 4, 4, 4], 
    depths=[3, 3, 5, 2], 
    num_stages=4,
    dec_outChannels=256, 
    drop_path=0.0,
    config=config
)

# Set device and move model to it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Initialize model utils and load checkpoint
metric = ConfusionMatrix(config['num_classes'])
mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.load_chkpt(model, None) # Don't need optimizer for evaluation

# Create evaluator
evaluator = Evaluator(model, metric)

# Evaluation loop
model.eval()
test_ious = []
test_bar = tqdm(test_loader)
print("Evaluating on test set...")

all_imgs = []
all_gts = []
all_preds = []

for step, test_batch in enumerate(test_bar):
    with torch.no_grad():
        evaluator.eval_step(test_batch)
        tiou = evaluator.get_scores()
        evaluator.reset_metric()
        
        # Get sample prediction for visualization
        img, gt, pred = evaluator.get_sample_prediction()
        all_imgs.append(img)
        all_gts.append(gt)
        all_preds.append(pred)
        
    test_ious.append(tiou['iou_mean'])
    test_bar.set_description(f'Test - t_mIOU {tiou["iou_mean"]:.4f}')

# Print overall test results
avg_test_iou = np.nanmean(test_ious)
print(f'=> Average Test IoU: {avg_test_iou:.4f}')

# Display several test predictions
num_samples = min(5, len(all_imgs))
plt.figure(figsize=(15, 3*num_samples))

for i in range(num_samples):
    plt.subplot(num_samples, 3, i*3+1)
    plt.imshow(all_imgs[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')
    
    plt.subplot(num_samples, 3, i*3+2)
    plt.imshow(g2c(all_gts[i])*255)
    plt.title(f"Ground Truth {i+1}")
    plt.axis('off')
    
    plt.subplot(num_samples, 3, i*3+3)
    plt.imshow(g2c(all_preds[i])*255)
    plt.title(f"Prediction {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show() 