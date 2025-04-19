#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
from pathlib import Path

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def check_gtfine_structure():
    """Check if gtFine has the expected directory structure."""
    required_dirs = ['train', 'val', 'test']
    for subdir in required_dirs:
        if not os.path.exists(f"gtFine/{subdir}"):
            print(f"Error: gtFine/{subdir} not found. Please ensure your gtFine directory has train, val, and test subdirectories.")
            return False
    return True

def download_leftimg8bit():
    """Download leftImg8bit dataset from Cityscapes website."""
    print("This script will download the leftImg8bit dataset (11GB+).")
    print("You need to have a Cityscapes account.")
    
    user_input = input("Do you want to download the leftImg8bit dataset? (y/n): ")
    if user_input.lower() != 'y':
        print("Please download the leftImg8bit dataset manually and place it in the current directory.")
        print("You can download it from: https://www.cityscapes-dataset.com/downloads/")
        return False
    
    username = input("Enter your Cityscapes username: ")
    password = input("Enter your Cityscapes password: ")
    
    # Download command
    cmd = [
        "wget", "--keep-session-cookies", "--save-cookies=cookies.txt", 
        "--post-data=username=" + username + "&password=" + password + "&submit=Login", 
        "https://www.cityscapes-dataset.com/login/"
    ]
    subprocess.run(cmd)
    
    # Download leftImg8bit dataset
    cmd = [
        "wget", "--load-cookies=cookies.txt", "-O", "leftImg8bit_trainvaltest.zip",
        "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
    ]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Error downloading the dataset. Please check your credentials or download it manually.")
        return False
    
    # Extract dataset
    print("Extracting dataset...")
    subprocess.run(["unzip", "leftImg8bit_trainvaltest.zip"])
    
    # Clean up
    os.remove("cookies.txt")
    
    return True

def prepare_directory_structure():
    """Create the proper directory structure for training."""
    # Create log and checkpoint directories
    create_directory("logs")
    create_directory("checkpoints")
    
    return True

def main():
    print("Setting up SegNeXt dataset...")
    
    # Check if gtFine directory has the expected structure
    if not check_gtfine_structure():
        return
    
    # Check if leftImg8bit exists, if not download it
    if not os.path.exists("leftImg8bit"):
        print("leftImg8bit directory not found.")
        if not download_leftimg8bit():
            print("Please set up the leftImg8bit dataset manually.")
            print("The directory structure should be:")
            print("- leftImg8bit/")
            print("  - train/")
            print("  - val/")
            print("  - test/")
    else:
        print("leftImg8bit directory found.")
    
    # Prepare directory structure
    prepare_directory_structure()
    
    print("\nSetup completed!")
    print("\nTo train the model, run:")
    print("python main.py")

if __name__ == "__main__":
    main() 