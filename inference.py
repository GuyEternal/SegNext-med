#!/usr/bin/env python3
import yaml
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import SegNext
from data_utils import std_norm, encode_labels, pallet_cityscape
from gray2color import gray2color
import argparse
from pathlib import Path

def load_config(config_path):
    with open(config_path) as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    return config

def load_model(config, checkpoint_path=None):
    # Create model
    model = SegNext(num_classes=config['num_classes'], 
                    in_channnels=config['input_channels'], 
                    embed_dims=[32, 64, 460, 256],
                    ffn_ratios=[4, 4, 4, 4], 
                    depths=[3, 3, 5, 2], 
                    num_stages=4,
                    dec_outChannels=256, 
                    drop_path=float(config['stochastic_drop_path']),
                    config=config)
    
    # Move to device                
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    model.eval()
    return model, device

def preprocess_image(image_path, config):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img, (config['img_width'], config['img_height']), cv2.INTER_LINEAR)
    
    # Normalize
    img_normalized = std_norm(img_resized)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0)
    
    return img, img_tensor

def inference(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        
        # Get the prediction
        if isinstance(output, tuple):
            output = output[0]  # Some models return additional outputs
            
        pred = output.data.max(1)[1].cpu().numpy()
        return pred[0]  # Return the first prediction (we only have one image)

def visualize_results(original_img, prediction, output_path=None):
    # Convert prediction to color using the cityscape palette
    colored_pred = gray2color(prediction, use_pallet='cityscape', custom_pallet=pallet_cityscape)
    
    # Create figure with original and prediction
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title("Segmentation Prediction")
    plt.imshow(colored_pred)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Results saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="SegNeXt inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to save output visualization")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model, device = load_model(config, args.checkpoint)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Preprocess image
    original_img, img_tensor = preprocess_image(args.image, config)
    
    # Run inference
    prediction = inference(model, img_tensor, device)
    
    # Visualize results
    output_path = args.output
    if output_path is None:
        # Generate output path based on input image if not specified
        img_path = Path(args.image)
        output_path = str(img_path.parent / f"{img_path.stem}_segmentation.png")
    
    visualize_results(original_img, prediction, output_path)

if __name__ == "__main__":
    main() 