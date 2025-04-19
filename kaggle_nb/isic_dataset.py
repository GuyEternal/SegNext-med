import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class TrainAugmentation:
    def __init__(self, img_size=256):
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def __call__(self, image, mask=None):
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask'].float()
        else:
            augmented = self.transform(image=image)
            return augmented['image']

class ValAugmentation:
    def __init__(self, img_size=256):
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def __call__(self, image, mask=None):
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask'].float()
        else:
            augmented = self.transform(image=image)
            return augmented['image']

class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, file_list=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        if file_list is None:
            # Get all image files
            self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        else:
            self.image_files = file_list
            
        self.image_files.sort()  # Ensure consistent ordering
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize to [0, 1]
        
        # Reshape mask to add channel dimension
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            # Default conversion to tensor if no transform provided
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        
        return img, mask

def get_data_loaders(data_dir, batch_size=16, train_transform=None, val_transform=None, val_split=0.2, seed=42):
    """
    Create training and validation data loaders
    
    Parameters:
    -----------
    data_dir : str
        Path to the ISIC dataset directory
    batch_size : int
        Batch size for data loaders
    train_transform : callable
        Transformations to apply to training data
    val_transform : callable
        Transformations to apply to validation data
    val_split : float
        Proportion of data to use for validation
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_loader, val_loader : DataLoader
        PyTorch data loaders for training and validation
    """
    # Set paths
    images_dir = os.path.join(data_dir, 'ISIC2018_Task1-2_Training_Input')
    masks_dir = os.path.join(data_dir, 'ISIC2018_Task1_Training_GroundTruth')
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"Masks directory not found: {masks_dir}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(
        image_files, test_size=val_split, random_state=seed
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets
    train_dataset = ISICDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
        file_list=train_files
    )
    
    val_dataset = ISICDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
        file_list=val_files
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader 