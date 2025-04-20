```yaml
# Add these parameters to config.yaml

# CrossNeXt Decoder Parameters
crossnext_num_heads: 8          # Number of attention heads for cross-axis attention
crossnext_kernel_sizes: [7, 11, 21]  # Kernel sizes for multi-scale strip convolutions
crossnext_norm_type: 'WithBias'      # Type of normalization in attention ('WithBias' or 'BiasFree')

# Note: The CrossNeXtDecoder will reuse the existing 'ham_channels' parameter if present
# in your config. If not, it will default to using the second encoder dimension value
# (enc_embed_dims[1]) which is typically 64.