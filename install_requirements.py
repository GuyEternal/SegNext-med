#!/usr/bin/env python3
import subprocess
import sys
import os

def install_packages():
    """Install required packages for the SegNeXt model."""
    packages = [
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "pyyaml",
        "matplotlib",
        "tqdm",
        "tabulate",
        "imgviz",
        "fmutils",
    ]
    
    print("Installing required packages...")
    
    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please install pip first.")
        return False
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}. Please install it manually.")
    
    # Create necessary directories
    create_directory("logs")
    create_directory("checkpoints")
    
    print("\nInstallation completed successfully!")
    return True

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    install_packages() 