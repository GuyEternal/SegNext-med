```python
# crossnext_decoder.py - Implementation of the Multi-Scale Cross-Axis Attention Decoder for SegNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from bricks import resize, ConvRelu  # Reuse functions from bricks.py

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Layer Normalization components
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def to_3d(x):
    """Convert input from 4D (B, C, H, W) to 3D (B, H*W, C) for layer normalization"""
    return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # B, C, H*W -> B, H*W, C

def to_4d(x, h, w):
    """Convert input from 3D (B, H*W, C) back to 4D (B, C, H, W) after layer normalization"""
    return x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)  # B, H*W, C -> B, C, H*W -> B, C, H, W

class BiasFree_LayerNorm(nn.Module):
    """Layer Normalization without bias term"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """Standard Layer Normalization with bias term"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """Layer Normalization wrapper that supports both with and without bias options"""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # Convert to 3D for normalization
        x_reshaped = to_3d(x)
        # Apply normalization
        x_normalized = self.body(x_reshaped)
        # Convert back to 4D
        return to_4d(x_normalized, h, w)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# CrossNeXt Attention Module
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CrossAxisAttention(nn.Module):
    """
    CrossNeXt Attention module - A multi-scale orthogonal axis attention mechanism.
    
    This module:
    1. Uses multi-scale strip convolutions on both x-axis and y-axis
    2. Applies cross-attention between the two branches
    3. Combines outputs with residual connections
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        LayerNorm_type (str): Type of layer normalization ('WithBias' or 'BiasFree')
        kernel_sizes (list): List of kernel sizes for multi-scale convolutions
    """
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias', kernel_sizes=[7, 11, 21]):
        super(CrossAxisAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        
        # Normalization layer
        self.norm = LayerNorm(dim, LayerNorm_type)
        
        # Multi-scale x-axis convolutions (horizontal strip convolutions)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, kernel_sizes[0]), 
                               padding=(0, kernel_sizes[0]//2), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, kernel_sizes[1]), 
                               padding=(0, kernel_sizes[1]//2), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, kernel_sizes[2]), 
                               padding=(0, kernel_sizes[2]//2), groups=dim)
        
        # Multi-scale y-axis convolutions (vertical strip convolutions)
        self.conv0_2 = nn.Conv2d(dim, dim, (kernel_sizes[0], 1), 
                               padding=(kernel_sizes[0]//2, 0), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (kernel_sizes[1], 1), 
                               padding=(kernel_sizes[1]//2, 0), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (kernel_sizes[2], 1), 
                               padding=(kernel_sizes[2]//2, 0), groups=dim)
        
        # Projection layers
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Multi-scale x-axis convolutions (horizontal)
        attn_0_1 = self.conv0_1(x_norm)
        attn_1_1 = self.conv1_1(x_norm)
        attn_2_1 = self.conv2_1(x_norm)
        
        # Combine x-axis outputs
        out1 = attn_0_1 + attn_1_1 + attn_2_1
        
        # Multi-scale y-axis convolutions (vertical)
        attn_0_2 = self.conv0_2(x_norm)
        attn_1_2 = self.conv1_2(x_norm)
        attn_2_2 = self.conv2_2(x_norm)
        
        # Combine y-axis outputs
        out2 = attn_0_2 + attn_1_2 + attn_2_2
        
        # Apply projections
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        
        # Reshape for cross attention
        # For x-axis branch query (using y-axis features)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        # For x-axis branch key and value (using x-axis features)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        
        # For y-axis branch query (using x-axis features)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        # For y-axis branch key and value (using y-axis features)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        
        # Normalize queries and keys for more stable attention
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        # Compute attention for x-axis branch
        attn1 = (q1 @ k1.transpose(-2, -1))  # B, num_heads, H, H
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1  # Add query as residual
        
        # Compute attention for y-axis branch
        attn2 = (q2 @ k2.transpose(-2, -1))  # B, num_heads, W, W
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2  # Add query as residual
        
        # Reshape back to 4D
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # Final projection and residual connection
        out3 = self.project_out(out3)
        out4 = self.project_out(out4)
        out = out3 + out4 + residual
        
        return out

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# CrossNeXt Decoder
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CrossNeXtDecoder(nn.Module):
    """
    CrossNeXt Decoder - An advanced orthogonal multi-scale attention decoder.
    
    This decoder:
    1. Takes features from multiple encoder stages
    2. Concatenates them after resizing to the same resolution
    3. Reduces channels with a 1x1 convolution
    4. Applies the CrossNeXt Attention mechanism
    5. Projects to output channels
    
    Args:
        outChannels (int): Number of output channels
        config (dict): Configuration dictionary
        enc_embed_dims (list): Dimensions of encoder features at each stage
    """
    def __init__(self, outChannels, config, enc_embed_dims=[32, 64, 460, 256]):
        super().__init__()
        
        # Get parameters from config
        intermediate_channels = config.get('ham_channels', enc_embed_dims[1])
        num_heads = config.get('crossnext_num_heads', 8)
        kernel_sizes = config.get('crossnext_kernel_sizes', [7, 11, 21])
        norm_type = config.get('crossnext_norm_type', 'WithBias')
        
        # Channel reduction (equivalent to "squeeze" in the original HamDecoder)
        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), intermediate_channels)
        
        # CrossNeXt Attention module
        self.cross_attn = CrossAxisAttention(
            dim=intermediate_channels, 
            num_heads=num_heads, 
            LayerNorm_type=norm_type,
            kernel_sizes=kernel_sizes
        )
        
        # Final output projection (equivalent to "align" in the original HamDecoder)
        self.align = ConvRelu(intermediate_channels, outChannels)
    
    def forward(self, features):
        # Drop stage 1 features (same as in the original HamDecoder)
        features = features[1:]
        
        # Resize all features to the same resolution (using the resolution of stage 3 features)
        # This matches the behavior in the original HamDecoder
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        
        # Concatenate features
        x = torch.cat(features, dim=1)
        
        # Channel reduction
        x = self.squeeze(x)
        
        # Apply CrossNeXt Attention
        x = self.cross_attn(x)
        
        # Final projection
        x = self.align(x)
        
        return x
```