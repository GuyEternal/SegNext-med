# Implementation Guide: Replacing HamDecoder with CrossNeXt Decoder

This guide outlines the steps to replace the current HamDecoder with the CrossNeXt Decoder in the SegNeXt model.

## Overview

The CrossNeXt decoder is designed as a drop-in replacement for the current HamDecoder while maintaining the same interface. It implements an orthogonal multi-scale attention mechanism that is more efficient and effective for medical image segmentation.

## Prerequisites

Before implementation, make sure you:
1. Understand the existing decoder structure in `decoder.py` and `hamburger.py`
2. Install the required `einops` library (see dependencies.md)
3. Have the appropriate permissions to modify the codebase

## Implementation Steps

### Step 1: Create the CrossNeXt Decoder File

Create a new file called `crossnext_decoder.py` with the implementation of the CrossNeXt decoder. This file includes:

- Layer normalization components
- CrossAxisAttention module
- CrossNeXtDecoder class

The CrossNeXtDecoder follows the same interface as the current HamDecoder:
- Takes encoder features
- Processes them (dropping stage 1 features)
- Resizes features to match the same resolution used in HamDecoder (features[-3].shape[2:])
- Returns the processed features for final classification

### Step 2: Update Configuration Files

Add the following parameters to the configuration file (`config.yaml`):

```yaml
# CrossNeXt Decoder Parameters
crossnext_num_heads: 8  # Number of attention heads
crossnext_kernel_sizes: [7, 11, 21]  # Kernel sizes for multi-scale convolutions
crossnext_norm_type: 'WithBias'  # Type of normalization in attention
```

The implementation will fallback to existing `ham_channels` value if provided in the config, maintaining backward compatibility.

### Step 3: Update Model File

Modify `model.py` to use the new CrossNeXtDecoder instead of HamDecoder:

1. Change the import statement:
   ```python
   # from decoder import HamDecoder
   from crossnext_decoder import CrossNeXtDecoder
   ```

2. Update the decoder initialization in the SegNext class:
   ```python
   self.decoder = CrossNeXtDecoder(
       outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
   ```

3. The forward method remains unchanged since the interface is the same.

## Key Components of CrossNeXt Decoder

### 1. Layer Normalization

These modules implement layer normalization for 4D tensors (batch, channels, height, width):
- `BiasFree_LayerNorm`: Layer normalization without bias
- `WithBias_LayerNorm`: Standard layer normalization with bias
- `LayerNorm`: Wrapper that allows choosing between the two types

### 2. CrossAxis Attention Module

The core component of the decoder, implementing an orthogonal multi-scale attention mechanism:

- **Multi-scale convolutions**:
  - Uses strip convolutions with different kernel sizes (7, 11, 21) for both x-axis and y-axis
  - Combines the outputs from different scales

- **Cross-axis attention**:
  - Reshapes features to compute attention along different spatial dimensions
  - Implements cross-attention between x-axis and y-axis features
  - Adds residual connections to improve gradient flow

### 3. CrossNeXtDecoder

The main decoder class that replaces HamDecoder:

1. Takes encoder features and drops the first stage features
2. Resizes all features to the same resolution (matching the original HamDecoder behavior)
3. Concatenates them along the channel dimension
4. Uses ConvRelu for channel reduction (matching the original squeeze)
5. Processes with the CrossAxisAttention module
6. Uses ConvRelu for final projection (matching the original align)

## Testing and Verification

After implementation, verify the following:

1. The model loads correctly without errors
2. The output shape matches the expected shape
3. The model runs through a forward pass successfully
4. The model can be trained (no NaN or infinity values)

## Compatibility Notes

- The CrossNeXtDecoder maintains the same interface as HamDecoder for seamless integration
- It reuses components from the existing codebase (ConvRelu, resize) to maintain consistency
- It uses the same resizing strategy as the original HamDecoder