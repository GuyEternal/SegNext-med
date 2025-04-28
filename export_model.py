import torch
from model import SegNext
import yaml

# Load config
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# Create model
model = SegNext(num_classes=config.get('num_classes', 19))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512)  # Adjust size based on your typical input size

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "segnext_model.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }
)

print("Model exported to segnext_model.onnx") 