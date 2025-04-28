# cross_attention.py - Implementation of the Multi-Scale Cross-Directional Attention Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

# Add proper error handling for einops import
try:
    from einops import rearrange
except ImportError:
    raise ImportError("Please install einops: pip install einops")

from bricks import resize, ConvRelu, DepthwiseSeparableConv  # Reuse and import functions from bricks.py

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
# Cross Attention Module
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CrossAttention(nn.Module):
    """
    CrossAttention module - A multi-scale cross-directional attention mechanism.
    
    This module:
    1. Uses multi-scale strip convolutions on both x-axis and y-axis
    2. Applies attention between the two orthogonal branches
    3. Combines outputs with residual connections
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        LayerNorm_type (str): Type of layer normalization ('WithBias' or 'BiasFree')
        kernel_sizes (list): List of kernel sizes for multi-scale convolutions
    """
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias', kernel_sizes=[7, 11, 21]):
        super(CrossAttention, self).__init__()
        
        # Add check to ensure that input dimension is divisible by number of heads
        assert dim % num_heads == 0, f"Input dimension {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        
        # Add temperature parameter as in CrossNet implementation
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Normalization layer - renamed to norm1 to match CrossNet
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
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
        x1 = self.norm1(x)
        
        # Multi-scale x-axis convolutions (horizontal) with CrossNet naming
        attn_00 = self.conv0_1(x1)
        attn_10 = self.conv1_1(x1)
        attn_20 = self.conv2_1(x1)
        
        # Combine x-axis outputs
        out1 = attn_00 + attn_10 + attn_20
        
        # Multi-scale y-axis convolutions (vertical) with CrossNet naming
        attn_01 = self.conv0_2(x1)
        attn_11 = self.conv1_2(x1)
        attn_21 = self.conv2_2(x1)
        
        # Combine y-axis outputs
        out2 = attn_01 + attn_11 + attn_21
        
        # Apply projections
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        
        # Reshape for cross attention
        # For the CrossNet implementation, we match the query, key, value assignments
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        
        # Normalize queries and keys for more stable attention
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        # Compute attention for x-axis branch
        # Attention weights: [batch, heads, height, height]
        attn1 = (q1 @ k1.transpose(-2, -1))  # B, num_heads, H, H
        attn1 = attn1.softmax(dim=-1)
        # Apply attention and add query as residual: [batch, heads, height, width*channels_per_head]
        out3 = (attn1 @ v1) + q1  # Add query as residual
        
        # Compute attention for y-axis branch
        # Attention weights: [batch, heads, width, width]
        attn2 = (q2 @ k2.transpose(-2, -1))  # B, num_heads, W, W
        attn2 = attn2.softmax(dim=-1)
        # Apply attention and add query as residual: [batch, heads, width, height*channels_per_head]
        out4 = (attn2 @ v2) + q2  # Add query as residual
        
        # Reshape back to 4D
        # [batch, heads, height, width*channels_per_head] -> [batch, channels, height, width]
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # [batch, heads, width, height*channels_per_head] -> [batch, channels, height, width]
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # Final projection and residual connection - simplified in CrossNet
        out = self.project_out(out3) + self.project_out(out4) + residual
        
        return out

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Cross Decoder
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CrossDecoder(nn.Module):
    """
    CrossDecoder - An advanced cross-directional multi-scale attention decoder.
    
    This decoder:
    1. Takes features from multiple encoder stages
    2. Concatenates them after resizing to the same resolution
    3. Reduces channels with a 1x1 convolution
    4. Applies the CrossAttention mechanism
    5. Projects to output channels
    
    Args:
        outChannels (int): Number of output channels
        config (dict): Configuration dictionary
        enc_embed_dims (list): Dimensions of encoder features at each stage
    """
    def __init__(self, outChannels, config, enc_embed_dims=[64, 128, 320, 512]):
        super().__init__()
        
        # Fixed-size parameter for consistent resizing
        self.image_size = config.get('decoder_image_size', 128)
        self.align_corners = config.get('align_corners', False)
        
        # Get parameters from config using the updated parameter names
        intermediate_channels = config.get('ham_channels', enc_embed_dims[1])
        
        # Add warning if ham_channels is not found in config
        if 'ham_channels' not in config:
            print(f"Warning: 'ham_channels' not found in config. Using encoder dimension {enc_embed_dims[1]} as fallback.")
            
        num_heads = config.get('cross_num_heads', 8)
        kernel_sizes = config.get('cross_kernel_sizes', [7, 11, 21])
        norm_type = config.get('cross_norm_type', 'WithBias')
        
        # Channel reduction for stages 1-3
        self.squeeze = ConvRelu(sum((enc_embed_dims[1], enc_embed_dims[2], enc_embed_dims[3])), 
                               intermediate_channels)
        
        # CrossAttention module
        self.decoder_level = CrossAttention(
            dim=intermediate_channels, 
            num_heads=num_heads, 
            LayerNorm_type=norm_type,
            kernel_sizes=kernel_sizes
        )
        
        # Add separable bottleneck for processing combined features
        # This is a key difference in the CrossNet implementation
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConv(
                intermediate_channels + enc_embed_dims[0],
                enc_embed_dims[3],
                kernel_size=3, 
                padding=1
            ),
            DepthwiseSeparableConv(
                enc_embed_dims[3],
                enc_embed_dims[3],
                kernel_size=3, 
                padding=1
            )
        )
        
        # Final output projection 
        self.align = ConvRelu(enc_embed_dims[3], outChannels)
        
        # Add classification head
        self.cls_seg = nn.Conv2d(outChannels, config.get('num_classes', 2), kernel_size=1)
    
    def forward(self, features):
        # Resize all features to fixed size instead of just dropping stage 1
        features = [resize(
            level, 
            size=(self.image_size, self.image_size), 
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in features]
        
        # Concatenate features from stages 1-3
        y1 = torch.cat([features[1], features[2], features[3]], dim=1)
        
        # Channel reduction
        x = self.squeeze(y1)
        
        # Apply CrossAttention
        x = self.decoder_level(x)
        
        # Concatenate with stage 0 features
        x = torch.cat([x, features[0]], dim=1)
        
        # Process through separable bottleneck
        x = self.sep_bottleneck(x)
        
        # Final projection
        x = self.align(x)
        
        # Apply classification head
        x = self.cls_seg(x)
        
        return x 