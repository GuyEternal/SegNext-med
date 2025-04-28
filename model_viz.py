import torch
from model import SegNext
import yaml
import os

# Load config
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# Create model
model = SegNext(num_classes=config.get('num_classes', 19))

# Option 1: Generate text summary with torchinfo
def generate_text_summary():
    try:
        from torchinfo import summary
        # Create a summary with a sample input size
        print("\n=== Model Summary with torchinfo ===")
        summary_str = summary(model, (1, 3, 512, 512), 
                              depth=3,  # Adjust depth as needed
                              col_names=["input_size", "output_size", "num_params", "kernel_size"],
                              verbose=0)
        
        # Write to file
        with open('model_summary.txt', 'w') as f:
            print(summary_str, file=f)
        print(f"Summary saved to model_summary.txt")
    except ImportError:
        print("torchinfo not installed. Install with: pip install torchinfo")

# Option 2: Generate TensorBoard visualization
def generate_tensorboard_viz():
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/model_architecture')
        writer.add_graph(model, torch.randn(1, 3, 512, 512))
        writer.close()
        print(f"\n=== TensorBoard graph created ===")
        print("Run 'tensorboard --logdir=runs' and open the URL in your browser")
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")

# Option 3: Generate a focused block diagram with graphviz
def generate_graphviz_diagram():
    try:
        import graphviz
        
        # Create a new directed graph
        dot = graphviz.Digraph(comment='SegNext Architecture with Detailed Decoder', format='png')
        dot.attr(rankdir='TB', size='11,16', dpi='300', bgcolor='white')
        
        # Create subgraph for overall structure
        with dot.subgraph(name='cluster_main') as main:
            main.attr(label='SegNext Model', style='rounded', color='black', fontsize='16', bgcolor='white')
            
            # Input
            main.node('input', 'Input\n(B×3×H×W)', shape='box', style='filled', fillcolor='lightblue', color='black')
            
            # Encoder - Simplified as a black box
            main.node('encoder', 'MSCANet Encoder\nEmbedding dims: [32, 64, 460, 256]\nDepths: [3, 3, 5, 2]', 
                      shape='box', style='filled', fillcolor='black', fontcolor='white')
            
            # Multi-scale feature outputs from encoder
            main.node('feature_s1', 'Stage 1 Features\n(H/4×W/4)', shape='box', style='filled', fillcolor='lightgrey')
            main.node('feature_s2', 'Stage 2 Features\n(H/8×W/8)', shape='box', style='filled', fillcolor='lightgrey')
            main.node('feature_s3', 'Stage 3 Features\n(H/16×W/16)', shape='box', style='filled', fillcolor='lightgrey')
            main.node('feature_s4', 'Stage 4 Features\n(H/32×W/32)', shape='box', style='filled', fillcolor='lightgrey')
            
            # Decoder - CrossNeXtDecoder with detailed structure
            with main.subgraph(name='cluster_decoder') as decoder:
                decoder.attr(label='CrossNeXtDecoder', style='rounded', color='darkblue', penwidth='2', bgcolor='lightyellow')
                
                # Feature concatenation and upsampling
                with decoder.subgraph(name='cluster_feature_fusion') as fusion:
                    fusion.attr(label='Feature Aggregation', style='rounded')
                    
                    # Upsampling paths
                    fusion.node('upsample_s4', 'Upsample\n(×2)', shape='box')
                    fusion.node('upsample_s3', 'Upsample\n(×2)', shape='box')
                    fusion.node('upsample_s2', 'Upsample\n(×2)', shape='box')
                    fusion.node('feature_concat', 'Feature Concatenation\n[S1, S2, S3, S4]\nFeature channels: 780', shape='box', style='filled', fillcolor='lightgoldenrod')
                
                # First ConvRelu (feature reduction)
                decoder.node('conv_relu1', 'ConvRelu\n1×1 Conv → ReLU\n780→512 channels', shape='box', style='filled', fillcolor='lightsalmon')
                
                # Cross-Axis Attention - Detailed module
                with decoder.subgraph(name='cluster_caa') as caa:
                    caa.attr(label='Cross-Axis Attention Module', style='rounded', color='darkred', penwidth='2')
                    
                    # LayerNorm
                    caa.node('layer_norm', 'LayerNorm', shape='box')
                    
                    # Parallel branches for horizontal attention
                    with caa.subgraph(name='cluster_h_attention') as h_attn:
                        h_attn.attr(label='Horizontal Attention', style='rounded')
                        h_attn.node('h_k7', '1×7 Conv\n512→512', shape='box')
                        h_attn.node('h_k11', '1×11 Conv\n512→512', shape='box')
                        h_attn.node('h_k21', '1×21 Conv\n512→512', shape='box')
                    
                    # Parallel branches for vertical attention
                    with caa.subgraph(name='cluster_v_attention') as v_attn:
                        v_attn.attr(label='Vertical Attention', style='rounded')
                        v_attn.node('v_k7', '7×1 Conv\n512→512', shape='box')
                        v_attn.node('v_k11', '11×1 Conv\n512→512', shape='box')
                        v_attn.node('v_k21', '21×1 Conv\n512→512', shape='box')
                    
                    # Feature aggregation
                    caa.node('h_concat', 'Concat H Features', shape='box')
                    caa.node('v_concat', 'Concat V Features', shape='box')
                    caa.node('hv_concat', 'Concat H+V Features', shape='box')
                    caa.node('caa_fusion', 'Feature Fusion\n1×1 Conv', shape='box')
                    caa.node('caa_add', 'Add', shape='diamond')
                
                # Final ConvRelu
                decoder.node('conv_relu2', 'ConvRelu\n1×1 Conv → ReLU\n512→256 channels', shape='box', style='filled', fillcolor='lightsalmon')
            
            # Classification head
            with main.subgraph(name='cluster_cls_head') as cls_head:
                cls_head.attr(label='Classification Head', style='rounded', color='darkgreen', penwidth='2', bgcolor='lightpink')
                cls_head.node('dropout', 'Dropout (p=0.1)', shape='box')
                cls_head.node('cls_conv', '1×1 Conv\n256→num_classes channels', shape='box')
            
            # Upsampling
            main.node('interp', 'Bilinear Interpolation\nto original size (H×W)', shape='box', style='filled', fillcolor='lightgreen')
            
            # Output
            main.node('output', 'Output\n(B×num_classes×H×W)', shape='box', style='filled', fillcolor='lightblue')
            
            # Connect main flow
            main.edge('input', 'encoder')
            
            # Connect encoder to multi-scale features
            main.edge('encoder', 'feature_s1', label='Stage 1')
            main.edge('encoder', 'feature_s2', label='Stage 2')
            main.edge('encoder', 'feature_s3', label='Stage 3')
            main.edge('encoder', 'feature_s4', label='Stage 4')
            
            # Connect features to decoder
            main.edge('feature_s1', 'feature_concat')
            main.edge('feature_s2', 'feature_concat')
            main.edge('feature_s3', 'upsample_s3')
            main.edge('feature_s4', 'upsample_s4')
            main.edge('upsample_s4', 'upsample_s3')
            main.edge('upsample_s3', 'upsample_s2')
            main.edge('upsample_s2', 'feature_concat')
            
            # Decoder internal flow
            main.edge('feature_concat', 'conv_relu1')
            main.edge('conv_relu1', 'layer_norm')
            
            # Cross-Axis Attention flow
            main.edge('layer_norm', 'h_k7')
            main.edge('layer_norm', 'h_k11')
            main.edge('layer_norm', 'h_k21')
            main.edge('layer_norm', 'v_k7')
            main.edge('layer_norm', 'v_k11')
            main.edge('layer_norm', 'v_k21')
            
            main.edge('h_k7', 'h_concat')
            main.edge('h_k11', 'h_concat')
            main.edge('h_k21', 'h_concat')
            main.edge('v_k7', 'v_concat')
            main.edge('v_k11', 'v_concat')
            main.edge('v_k21', 'v_concat')
            
            main.edge('h_concat', 'hv_concat')
            main.edge('v_concat', 'hv_concat')
            main.edge('hv_concat', 'caa_fusion')
            main.edge('caa_fusion', 'caa_add')
            main.edge('layer_norm', 'caa_add', style='dashed')  # Skip connection
            main.edge('caa_add', 'conv_relu2')
            
            # Classification head flow
            main.edge('conv_relu2', 'dropout')
            main.edge('dropout', 'cls_conv')
            main.edge('cls_conv', 'interp')
            main.edge('interp', 'output')
        
        # Save and render the graph
        dot.render('model_architecture_decoder_focus', view=False)
        print(f"\n=== Decoder-focused diagram created ===")
        print("Saved as model_architecture_decoder_focus.png")
    except ImportError:
        print("Graphviz not installed. Install with: pip install graphviz")

if __name__ == "__main__":
    print("Generating visualizations for SegNext model...")
    
    # Run visualization method
    generate_graphviz_diagram()
    
    print("\nDone! Decoder-focused diagram created.") 