# SegNeXt: Semantic Segmentation Setup Guide

This guide explains how to set up the SegNeXt semantic segmentation model with the Cityscapes dataset.

## Dataset Structure

The Cityscapes dataset consists of two main parts:
1. **gtFine**: Contains the semantic segmentation labels (already available in your repository)
2. **leftImg8bit**: Contains the original RGB images (needs to be downloaded)

Each part should have the following structure:
```
gtFine/
├── train/
│   ├── aachen/
│   ├── bochum/
│   └── ...
├── val/
│   ├── frankfurt/
│   ├── lindau/
│   └── ...
└── test/
    ├── berlin/
    ├── bielefeld/
    └── ...

leftImg8bit/
├── train/
│   ├── aachen/
│   ├── bochum/
│   └── ...
├── val/
│   ├── frankfurt/
│   ├── lindau/
│   └── ...
└── test/
    ├── berlin/
    ├── bielefeld/
    └── ...
```

## Setup Steps

1. **Run the setup script**:
   ```
   python setup_dataset.py
   ```
   This script will:
   - Check if the gtFine directory has the proper structure
   - Help you download the leftImg8bit dataset if needed
   - Create log and checkpoint directories

2. **Download the leftImg8bit dataset**:
   - If you don't want to use the automated script, you can download it manually from [Cityscapes website](https://www.cityscapes-dataset.com/downloads/)
   - You need to register for an account on the website
   - Download the "leftImg8bit_trainvaltest.zip" file (11GB+)
   - Extract it in the repository root directory

## Training the Model

1. **Adjust the configuration** (if needed):
   - Edit `config.yaml` to change training parameters
   - Adjust batch size, learning rate, etc. based on your hardware

2. **Start training**:
   ```
   python main.py
   ```

3. **Monitor training**:
   - Training progress will be displayed in the console
   - Logs will be saved in the `logs` directory
   - Checkpoints will be saved in the `checkpoints` directory

## Model Details

SegNeXt is a convolutional attention-based model for semantic segmentation that:
- Uses an efficient convolutional attention mechanism
- Outperforms many transformer-based models
- Has better performance-to-computation ratio than previous state-of-the-art methods

The implementation includes:
- A backbone network with hierarchical feature extraction
- A hamburger module for efficient context modeling
- A lightweight decoder for upsampling the features

## Hardware Requirements

- GPU with at least 8GB VRAM (recommended)
- If you have memory issues, reduce the batch size in `config.yaml`
- For multi-GPU training, change `gpus_to_use` in `config.yaml` and set `norm_typ` to `sync_bn`

## Troubleshooting

- **Memory issues**: Reduce batch size or image resolution in `config.yaml`
- **Missing dataset**: Follow the download instructions above
- **Training is slow**: Consider reducing the image resolution or using a more powerful GPU 