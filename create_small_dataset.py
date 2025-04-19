#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path
import argparse

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def create_small_dataset(source_dir, target_dir, num_samples=10, seed=42):
    """
    Create a small subset of the Cityscapes dataset for quick testing.
    
    Args:
        source_dir: Original dataset directory
        target_dir: Directory to save the small dataset
        num_samples: Number of samples to include per split (train/val/test)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Define the structure
    splits = ['train', 'val', 'test']
    sub_dirs = ['leftImg8bit', 'gtFine']
    
    # Create target directory structure
    for sub_dir in sub_dirs:
        for split in splits:
            create_directory(os.path.join(target_dir, sub_dir, split))
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get all city directories in the leftImg8bit split directory
        img_split_dir = os.path.join(source_dir, 'leftImg8bit', split)
        if not os.path.exists(img_split_dir):
            print(f"Warning: {img_split_dir} does not exist, skipping.")
            continue
        
        city_dirs = [d for d in os.listdir(img_split_dir) if os.path.isdir(os.path.join(img_split_dir, d))]
        
        # Track statistics
        total_img_copied = 0
        total_label_copied = 0
        
        # Process each city
        for city in city_dirs:
            print(f"  Processing city: {city}")
            
            # Create city directories in target
            create_directory(os.path.join(target_dir, 'leftImg8bit', split, city))
            create_directory(os.path.join(target_dir, 'gtFine', split, city))
            
            # Get all image files for this city
            img_city_dir = os.path.join(source_dir, 'leftImg8bit', split, city)
            img_files = [f for f in os.listdir(img_city_dir) if f.endswith('_leftImg8bit.png')]
            
            # Select random subset
            if len(img_files) > num_samples:
                selected_img_files = random.sample(img_files, num_samples)
            else:
                selected_img_files = img_files
                print(f"    Warning: Only {len(img_files)} files available in {split}/{city}")
            
            # Copy selected files
            for img_file in selected_img_files:
                # Copy image file
                src_img_path = os.path.join(source_dir, 'leftImg8bit', split, city, img_file)
                dst_img_path = os.path.join(target_dir, 'leftImg8bit', split, city, img_file)
                
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    total_img_copied += 1
                    print(f"    Copied {src_img_path} to {dst_img_path}")
                else:
                    print(f"    Warning: Source image {src_img_path} does not exist")
                
                # Find and copy corresponding label file
                # Convert: city_000123_000456_leftImg8bit.png -> city_000123_000456_gtFine_labelIds.png
                base_name = img_file.replace('_leftImg8bit.png', '')
                label_file = f"{base_name}_gtFine_labelIds.png"
                
                src_label_path = os.path.join(source_dir, 'gtFine', split, city, label_file)
                dst_label_path = os.path.join(target_dir, 'gtFine', split, city, label_file)
                
                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dst_label_path)
                    total_label_copied += 1
                    print(f"    Copied {src_label_path} to {dst_label_path}")
                else:
                    print(f"    Warning: Source label {src_label_path} does not exist")
        
        print(f"  {split}: Copied {total_img_copied} images and {total_label_copied} labels")
    
    print(f"\nSmall dataset created at {target_dir}")
    print("Directory structure:")
    print(f"  {target_dir}/")
    print(f"    leftImg8bit/")
    print(f"      train/ [city directories with images]")
    print(f"      val/ [city directories with images]")
    print(f"      test/ [city directories with images]")
    print(f"    gtFine/")
    print(f"      train/ [city directories with labels]")
    print(f"      val/ [city directories with labels]")
    print(f"      test/ [city directories with labels]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a small subset of the Cityscapes dataset for testing")
    parser.add_argument("--source", type=str, default="./dataset", help="Source dataset directory")
    parser.add_argument("--target", type=str, default="./small_dataset", help="Target directory for small dataset")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples per city per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_small_dataset(args.source, args.target, args.samples, args.seed)
