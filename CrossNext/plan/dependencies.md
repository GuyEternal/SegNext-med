# Dependencies for CrossNeXt Implementation

## Required Python Packages

The CrossNeXt implementation requires the `einops` library, which provides convenient tensor operations for deep learning. This library is the only additional dependency not already present in the existing codebase.

### Installation

```bash
# Option 1: Install with pip directly
pip install einops

# Option 2: Add to requirements.txt and install
# Add this line to requirements.txt:
# einops>=0.6.0
# Then run:
# pip install -r requirements.txt
```

### Compatibility Notes

- The `einops` library is lightweight and compatible with PyTorch
- Tested with `einops` version 0.6.0 and above
- No other dependencies need to be added for the CrossNeXt implementation as it leverages existing components from the codebase