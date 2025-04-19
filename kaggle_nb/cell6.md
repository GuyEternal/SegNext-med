```python
# Update dataloader.py to handle ISIC dataset structure
%%writefile dataloader.py
from cv2 import INTER_NEAREST
import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.utils.data as data
from fmutils import fmutils as fmu
from tabulate import tabulate
import cv2
import numpy as np
import os, random, time

from data_utils import std_norm, encode_labels

class GEN_DATA_LISTS():
    
    def __init__(self, root_dir, sub_dirname):
        '''
        Parameters
        ----------
        root_dir : TYPE
            root directory containing [train, test, val] folders.
        sub_dirname : TYPE
            sub directories inside the main split (train, test, val) folders
        get_lables_from : TYPE
            where to get the label from either from dir_name of file_name.
        '''
        self.root_dir = root_dir
        self.sub_dirname = sub_dirname
        
        # Use different splits based on dataset type
        if 'small_isic_dataset' in root_dir or config['num_classes'] == 2:
            # ISIC dataset splits
            self.splits = ['Training', 'Validation', 'Test_v2']
        else:
            # Original Cityscapes splits
            self.splits = ['train', 'val', 'test']
        
    def get_splits(self):
        
        print('Directories loaded:')
        self.split_files = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[0]))
            self.split_files.append(os.path.join(self.root_dir, split, self.sub_dirname[0]))
        print('\n')
        self.split_lbls = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            self.split_lbls.append(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            
        # Create simple utility function for file collection
        def get_all_files(path):
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist, returning empty list.")
                return []
            return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            
        # Get all files from each directory
        self.train_f = get_all_files(self.split_files[0])
        self.val_f = get_all_files(self.split_files[1])
        self.test_f = get_all_files(self.split_files[2])

        self.train_l = get_all_files(self.split_lbls[0])
        self.val_l = get_all_files(self.split_lbls[1])
        self.test_l = get_all_files(self.split_lbls[2])
        
        train, val, test = [self.train_f, self.train_l], [self.val_f, self.val_l], [self.test_f, self.test_l]
        
        return train, val, test
    
    def get_classes(self):
        if config['num_classes'] == 2:  # ISIC dataset
            return ['background', 'lesion']
        else:  # Cityscapes dataset
            cls_names = []
            for i in range(len(self.train_f)):
                cls_names.append(os.path.basename(self.train_f[i]).split('_')[1])
            classes = sorted(list(set(cls_names)))
            return classes
    
    def get_filecounts(self):
        print('\n')
        result = np.concatenate((np.asarray(self.splits).reshape(-1,1),
                                np.asarray([len(self.train_f), len(self.val_f), len(self.test_f)]).reshape(-1,1),
                                np.asarray([len(self.train_l), len(self.val_l), len(self.test_l)]).reshape(-1,1))
         , 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Split", "Images", "Labels"], tablefmt="github"))
        return None

class ISICDataset(data.Dataset):
    def __init__(self, img_paths, mask_paths, img_height, img_width, augment_data=False, normalize=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_height = img_height
        self.img_width = img_width
        self.augment_data = augment_data
        self.normalize = normalize
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        data_sample = {}
        
        # Load image and resize to standard dimensions
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height), cv2.INTER_LINEAR)

        # Get mask path by parsing image filename to match mask naming pattern
        img_filename = os.path.basename(self.img_paths[index])
        mask_filename = os.path.splitext(img_filename)[0] + "_segmentation.png"
        mask_path = os.path.join(os.path.dirname(self.mask_paths[index]), mask_filename)
        
        # Load mask and resize
        lbl = cv2.imread(mask_path, 0)
        lbl = cv2.resize(lbl, (self.img_width, self.img_height), cv2.INTER_NEAREST)
        lbl = encode_labels(lbl)  # This will now handle binary encoding

        if self.augment_data:
            # Data augmentation can be implemented here
            pass
            
        if self.normalize:
            img = std_norm(img)
        
        data_sample['img'] = img
        data_sample['lbl'] = lbl

        return data_sample

# Create a simple utility module for file operations
class fmutils:
    @staticmethod
    def get_all_files(path):
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, returning empty list.")
            return []
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    @staticmethod
    def get_basename(path):
        return os.path.basename(path)
``` 