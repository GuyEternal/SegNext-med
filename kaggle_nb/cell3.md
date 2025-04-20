# Cell 3: Verify dataset structure

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Display dataset structure
print("Dataset Structure:")
!ls -la /kaggle/input/processed-dataset/processed_isic_dataset

# Check for required subdirectories
required_dirs = [
    '/kaggle/input/processed-dataset/processed_isic_dataset/Training', 
    '/kaggle/input/processed-dataset/processed_isic_dataset/Training/images', 
    '/kaggle/input/processed-dataset/processed_isic_dataset/Training/masks'
]

all_exist = True
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path} exists")
        # Count files
        files = os.listdir(dir_path)
        print(f"  - Contains {len(files)} files")
    else:
        print(f"✗ {dir_path} does not exist")
        all_exist = False

if not all_exist:
    print("\nWARNING: Some required directories are missing. Dataset structure should be:")
    print("/kaggle/input/processed-dataset/processed_isic_dataset")
    print("├── Training/")
    print("│   ├── images/")
    print("│   └── masks/")
    print("├── Validation/")
    print("│   ├── images/")
    print("│   └── masks/")
    print("└── Test_v2/")
    print("    ├── images/")
    print("    └── masks/")

# Display a sample image and mask if available
if all_exist:
    # Get a sample image
    img_path = os.path.join('/kaggle/input/processed-dataset/processed_isic_dataset/Training/images', os.listdir('/kaggle/input/processed-dataset/processed_isic_dataset/Training/images')[0])
    mask_path = os.path.join('/kaggle/input/processed-dataset/processed_isic_dataset/Training/masks', os.listdir('/kaggle/input/processed-dataset/processed_isic_dataset/Training/masks')[0])
    
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('Sample Image')
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Sample Mask')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show() 