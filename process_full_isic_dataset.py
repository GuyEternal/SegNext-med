#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import argparse
import sys

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def process_full_isic_dataset(source_dir, target_dir, force=False):
    """
    Process the full ISIC-2017 dataset, copying images with valid masks.
    
    Args:
        source_dir: Original dataset directory
        target_dir: Directory to save the processed dataset
        force: If True, overwrite existing target directory without asking
    """
    # Check if target directory exists and is not empty
    if os.path.exists(target_dir) and os.listdir(target_dir) and not force:
        response = input(f"Target directory {target_dir} is not empty. Do you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Aborting.")
            sys.exit(1)
    
    # Define the structure
    splits = ['Training', 'Validation', 'Test_v2']
    sub_dirs = ['images', 'masks']
    
    # Define paths
    source_paths = {
        'Training': {
            'images': os.path.join(source_dir, 'ISIC-2017_Training_Data', 'ISIC-2017_Training_Data'),
            'masks': os.path.join(source_dir, 'ISIC-2017_Training_Part1_GroundTruth', 'ISIC-2017_Training_Part1_GroundTruth')
        },
        'Validation': {
            'images': os.path.join(source_dir, 'ISIC-2017_Validation_Data', 'ISIC-2017_Validation_Data'),
            'masks': os.path.join(source_dir, 'ISIC-2017_Validation_Part1_GroundTruth', 'ISIC-2017_Validation_Part1_GroundTruth')
        },
        'Test_v2': {
            'images': os.path.join(source_dir, 'ISIC-2017_Test_v2_Data', 'ISIC-2017_Test_v2_Data'),
            'masks': os.path.join(source_dir, 'ISIC-2017_Test_v2_Part1_GroundTruth', 'ISIC-2017_Test_v2_Part1_GroundTruth')
        }
    }
    
    # Create target directory structure
    for split in splits:
        for sub_dir in sub_dirs:
            create_directory(os.path.join(target_dir, split, sub_dir))
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get paths for this split
        img_dir = source_paths[split]['images']
        mask_dir = source_paths[split]['masks']
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist, skipping.")
            continue
            
        if not os.path.exists(mask_dir):
            print(f"Warning: {mask_dir} does not exist, skipping.")
            continue
        
        # Get all image files (case-insensitive, ignore superpixels and metadata)
        img_files = [f for f in os.listdir(img_dir) 
                   if f.lower().endswith('.jpg') 
                   and not '_superpixels' in f]
        
        # Filter to only include images that have corresponding masks
        valid_img_files = []
        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_segmentation.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                valid_img_files.append(img_file)
            else:
                print(f"    Warning: No mask found for {img_file}, skipping")
        
        print(f"  Found {len(valid_img_files)} images with valid masks out of {len(img_files)} total images")
        
        # Track statistics
        total_img_copied = 0
        total_mask_copied = 0
        
        # Copy all valid files
        for i, img_file in enumerate(valid_img_files, 1):
            # Get the base name (e.g., ISIC_0000123 from ISIC_0000123.jpg)
            base_name = os.path.splitext(img_file)[0]
            
            # Define the mask file name (e.g., ISIC_0000123_segmentation.png)
            mask_file = f"{base_name}_segmentation.png"
            
            # Copy image file
            src_img_path = os.path.join(img_dir, img_file)
            dst_img_path = os.path.join(target_dir, split, 'images', img_file)
            
            shutil.copy2(src_img_path, dst_img_path)
            total_img_copied += 1
            
            # Copy mask file
            src_mask_path = os.path.join(mask_dir, mask_file)
            dst_mask_path = os.path.join(target_dir, split, 'masks', mask_file)
            
            shutil.copy2(src_mask_path, dst_mask_path)
            total_mask_copied += 1
            
            # Print progress periodically (every 10 files)
            if i % 10 == 0 or i == len(valid_img_files):
                print(f"    Progress: {i}/{len(valid_img_files)} files processed")
        
        print(f"  {split}: Copied {total_img_copied} images and {total_mask_copied} masks")
    
    print(f"\nFull dataset processed and saved at {target_dir}")
    print("Directory structure:")
    print(f"  {target_dir}/")
    for split in splits:
        print(f"    {split}/")
        print(f"      images/ [contains jpg images]")
        print(f"      masks/ [contains png segmentation masks]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the full ISIC-2017 dataset")
    parser.add_argument("--source", type=str, default="./dataset/isic-2017-kaggle", help="Source dataset directory")
    parser.add_argument("--target", type=str, default="./processed_isic_dataset", help="Target directory for processed dataset")
    parser.add_argument("--force", action="store_true", help="Force overwrite of target directory without asking")
    
    args = parser.parse_args()
    
    process_full_isic_dataset(args.source, args.target, args.force) 