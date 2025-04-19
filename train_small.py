#%%
import yaml
with open('config_small.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp # Add multiprocessing support

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300

from dataloader import GEN_DATA_LISTS, Cityscape
from data_utils import collate, pallet_cityscape, torch_resizer, masks_transform
from model import SegNext
from losses import FocalLoss, CrossEntropyLoss2d
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer, Evaluator, ModelUtils
import torch.nn.functional as F

from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='cityscape', custom_pallet=pallet_cityscape)

# Add debug wrapper for model forward method
class ModelDebugWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        print(f"\nInput shape: {x.shape}")
        try:
            # Get encoder features
            enc_feats = self.model.encoder(x)
            print(f"Encoder features shapes: {[f.shape for f in enc_feats]}")
            
            # Trace decoder
            dec_out = self.model.decoder(enc_feats)
            print(f"Decoder output shape: {dec_out.shape}")
            
            # Final conv and upsampling
            output = self.model.cls_conv(dec_out)
            print(f"Classifier output shape: {output.shape}")
            
            output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)
            print(f"Final output shape: {output.shape}")
            
            return output
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            # Add detailed traceback
            import traceback
            traceback.print_exc()
            raise e

# Function to execute the training process
def main():
    # Check if small dataset exists
    if not os.path.exists(config['data_dir']):
        print(f"Small dataset directory {config['data_dir']} not found.")
        print("Run 'python create_small_dataset.py' first to create a small test dataset.")
        exit(1)

    print(f"Loading data from {config['data_dir']}")

    # Modified to directly build file lists instead of using GEN_DATA_LISTS
    # since the directory structure doesn't match what GEN_DATA_LISTS expects
    import glob
    from pathlib import Path

    def get_file_lists(data_dir, subdir):
        file_lists = []
        for split in ['train', 'val', 'test']:
            files = []
            # Get all city directories
            city_dirs = glob.glob(os.path.join(data_dir, subdir, split, '*'))
            # Get all files in each city directory
            for city_dir in city_dirs:
                city_files = glob.glob(os.path.join(city_dir, '*.png'))
                files.extend(city_files)
            file_lists.append(sorted(files))
        return file_lists

    # Get image files
    img_files = get_file_lists(config['data_dir'], config['sub_directories'][0])
    train_img, val_img, test_img = img_files

    # Get label files - convert image filenames to corresponding label filenames
    def convert_to_label_path(img_path, img_subdir, lbl_subdir):
        # Replace leftImg8bit with gtFine in the path
        rel_path = os.path.relpath(img_path, os.path.join(config['data_dir'], img_subdir))
        # Get components: split/city/filename
        components = rel_path.split(os.sep)
        # Replace filename suffix
        filename = components[-1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        components[-1] = filename
        # Build new path
        new_rel_path = os.path.join(*components)
        return os.path.join(config['data_dir'], lbl_subdir, new_rel_path)

    train_lbl = [convert_to_label_path(p, config['sub_directories'][0], config['sub_directories'][1]) for p in train_img]
    val_lbl = [convert_to_label_path(p, config['sub_directories'][0], config['sub_directories'][1]) for p in val_img]
    test_lbl = [convert_to_label_path(p, config['sub_directories'][0], config['sub_directories'][1]) for p in test_img]

    # Print file counts
    print('\n')
    print("| Split   |   Images |   Labels |")
    print("|---------|----------|----------|")
    print(f"| train   |     {len(train_img)} |     {len(train_lbl)} |")
    print(f"| val     |     {len(val_img)} |     {len(val_lbl)} |")
    print(f"| test    |     {len(test_img)} |     {len(test_lbl)} |")

    train_paths = [train_img, train_lbl]
    val_paths = [val_img, val_lbl]
    test_paths = [test_img, test_lbl]

    # Reduce num_workers to 0 on macOS to avoid multiprocessing issues
    if sys.platform == 'darwin':  # macOS
        config['num_workers'] = 0
        print("Running on macOS, setting num_workers=0 to avoid multiprocessing issues")

    # Create train dataset and loader
    print("Creating training data loader...")
    train_data = Cityscape(train_paths[0], train_paths[1], config['img_height'], config['img_width'],
                        config['Augment_data'], config['Normalize_data'])

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['num_workers'],
                            collate_fn=collate, pin_memory=config['pin_memory'])

    # Create validation dataset and loader
    print("Creating validation data loader...")
    val_data = Cityscape(val_paths[0], val_paths[1], config['img_height'], config['img_width'],
                        False, config['Normalize_data'])

    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'],
                            collate_fn=collate, pin_memory=config['pin_memory'])

    # DataLoader Sanity Checks
    print("Checking data batch...")
    try:
        batch = next(iter(train_loader))
        s=255
        img_ls = []
        [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(min(config['batch_size'], len(batch['img'])))]
        [img_ls.append(g2c(batch['lbl'][i])) for i in range(min(config['batch_size'], len(batch['lbl'])))]
        plt.figure(figsize=(10, 6))
        plt.title('Sample Batch')
        plt.imshow(imgviz.tile(img_ls, shape=(2, min(config['batch_size'], len(batch['img']))), border=(255,0,0)))
        plt.axis('off')
        plt.savefig('sample_batch.png')
        print("Sample batch visualization saved as 'sample_batch.png'")
    except Exception as e:
        print(f"Error during batch visualization: {e}")

    print("Creating model...")
    # Initialize model with smaller dimensions for testing
    # IMPORTANT: Removed dropout parameter which was causing the error
    model = SegNext(num_classes=config['num_classes'], in_channnels=3, embed_dims=[32, 64, 160, 256],
                    ffn_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2], num_stages=4,
                    dec_outChannels=128, drop_path=float(config['stochastic_drop_path']),
                    config=config)
    
    # We'll use our custom wrapper during debugging only
    if config.get('debug_mode', False):
        model = ModelDebugWrapper(model)
                    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    print(f"Model created and moved to {device}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    loss = FocalLoss()
    criterion = lambda x,y: loss(x, y)

    optimizer = torch.optim.AdamW([{'params': model.parameters(),
                                'lr':config['learning_rate']}], weight_decay=0.0005)

    scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                            iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

    metric = ConfusionMatrix(config['num_classes'])

    mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
    # Comment this out if you don't have a checkpoint to load
    # mu.load_chkpt(model, optimizer)

    # Custom image transform function for converting NumPy arrays to properly formatted PyTorch tensors
    def custom_images_transform(images):
        inputs = []
        for img in images:
            # Convert from (H, W, C) to (C, H, W)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            inputs.append(img_tensor)
        inputs = torch.stack(inputs, dim=0).to(device)
        return inputs

    # Override the training and evaluation step methods to use our custom transform
    class CustomTrainer(Trainer):
        def training_step(self, batched_data):
            img_batch = custom_images_transform(batched_data['img'])
            lbl_batch = masks_transform(batched_data['lbl'])
            
            self.optimizer.zero_grad()
            preds = self.model.forward(img_batch)
            
            # Ensure predictions and targets are the same size
            if preds.size()[2:] != lbl_batch.size()[1:]:
                # Either downsample the target or upsample the predictions
                lbl_batch = F.interpolate(lbl_batch.float().unsqueeze(1), size=preds.size()[2:], 
                                         mode='nearest').long().squeeze(1)
            
            loss = self.criterion(preds, lbl_batch)
            loss.backward()
            self.optimizer.step()
            
            preds = preds.argmax(1)
            preds = preds.cpu().numpy()
            lbl_batch = lbl_batch.cpu().numpy()
            
            self.metric.update(lbl_batch, preds)
            return loss.item()

    class CustomEvaluator(Evaluator):
        def eval_step(self, data_batch):
            self.img_batch = custom_images_transform(data_batch['img'])
            lbl_batch = masks_transform(data_batch['lbl'])
            
            with torch.no_grad():
                preds = self.model.forward(self.img_batch)
                
                # Ensure predictions and targets are the same size
                if preds.size()[2:] != lbl_batch.size()[1:]:
                    # Either downsample the target or upsample the predictions
                    lbl_batch = F.interpolate(lbl_batch.float().unsqueeze(1), size=preds.size()[2:], 
                                             mode='nearest').long().squeeze(1)
            
            preds = preds.argmax(1)
            self.preds = preds.cpu().numpy()
            self.lbl_batch = lbl_batch.cpu().numpy()
            self.metric.update(self.lbl_batch, self.preds)

    # Test forward pass with a single batch before full training
    print("\nTesting forward pass with a single batch...")
    try:
        test_batch = next(iter(train_loader))
        test_input = custom_images_transform(test_batch['img'])
        print(f"Test input shape: {test_input.shape}")
        with torch.no_grad():
            _ = model(test_input)
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error during test forward pass: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit if test forward pass fails

    trainer = CustomTrainer(model, config['batch_size'], optimizer, criterion, metric)
    evaluator = CustomEvaluator(model, metric)

    print(f"Starting training for {config['epochs']} epochs...")
    epoch, best_iou, curr_viou = 0, 0, 0
    total_avg_viou = []

    try:
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            
            # Training phase
            pbar = tqdm(train_loader)
            model.train()
            ta, tl = [], []
            for step, data_batch in enumerate(pbar):
                scheduler(optimizer, step, epoch)
                loss_value = trainer.training_step(data_batch)
                iou = trainer.get_scores()
                trainer.reset_metric()
                
                tl.append(loss_value)
                ta.append(iou['iou_mean'])
                pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
            
            print(f'=> Average loss: {np.nanmean(tl)}, Average IoU: {np.nanmean(ta)}')

            # Validation phase (every epoch for small dataset)
            model.eval()
            va = []
            vbar = tqdm(val_loader)
            for step, val_batch in enumerate(vbar):
                with torch.no_grad():
                    evaluator.eval_step(val_batch)
                    viou = evaluator.get_scores()
                    evaluator.reset_metric()

                va.append(viou['iou_mean'])
                vbar.set_description(f'Validation - v_mIOU {viou["iou_mean"]:.4f}')

            # Generate a sample prediction for visualization
            try:
                img, gt, pred = evaluator.get_sample_prediction()
                tiled = imgviz.tile([img, g2c(gt), g2c(pred)], shape=(1,3), border=(255,0,0))
                plt.figure(figsize=(12, 4))
                plt.imshow(tiled)
                plt.axis('off')
                plt.savefig(f'prediction_epoch_{epoch+1}.png')
                print(f"Prediction visualization saved as 'prediction_epoch_{epoch+1}.png'")
            except Exception as e:
                print(f"Error during prediction visualization: {e}")
                
            avg_viou = np.nanmean(va)
            total_avg_viou.append(avg_viou)
            curr_viou = np.nanmax(total_avg_viou)
            print(f'=> Averaged Validation IoU: {avg_viou:.4f}')
            
            # Save model if improved
            if curr_viou > best_iou:
                best_iou = curr_viou
                mu.save_chkpt(model, optimizer, epoch, np.nanmean(tl), best_iou)
                print(f"=> Model saved with improved validation IoU: {best_iou:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTraining completed. Best validation IoU: {best_iou:.4f}")

# Add the necessary imports for proper multiprocessing
if __name__ == '__main__':
    import sys
    # Fix for macOS multiprocessing issues
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    main()
