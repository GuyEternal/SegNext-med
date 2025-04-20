#!/bin/bash

# Set path to checkpoint
CHECKPOINT="/kaggle/input/checkpoint_till_train/isic_segmentation_model_after_train_benchmark_on_isic2017_segnextbasemodel.pth"
CONFIG="../config_kaggle.yaml"
OUTPUT_DIR="test_results"

# Make script executable
chmod +x test_model.py

# Run the test script
python test_model.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --output_dir "$OUTPUT_DIR" 