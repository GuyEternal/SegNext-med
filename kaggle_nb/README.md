# Running SegNext-med on Kaggle Free Tier

This directory contains code cells for running the SegNext-med semantic segmentation model on Kaggle's free tier. The code is organized into cells that can be copied directly into a Kaggle notebook.

## Dataset Requirements

You need to upload the `small_isic_dataset` to Kaggle as a dataset with the following structure:

```
small_isic_dataset/
├── Training/
│   ├── images/ (contains image files)
│   └── masks/ (contains mask files)
├── Validation/
│   ├── images/
│   └── masks/
└── Test_v2/
    ├── images/
    └── masks/
```

## How to Use

1. Create a new Kaggle notebook
2. Add the `small_isic_dataset` as an input dataset to your notebook
3. Copy the contents of each `.md` file into separate code cells in your Kaggle notebook
4. Run the cells in sequential order:

- **cell1.md**: Clones the repository and installs dependencies
- **cell2.md**: Creates necessary directories and configures the model for Kaggle
- **cell3.md**: Verifies the dataset structure
- **cell4.md**: Imports modules and prepares for training
- **cell5.md**: Creates data loaders for training and validation
- **cell6.md**: Initializes the SegNext model
- **cell7.md**: Runs the training loop
- **cell8.md**: Loads the best model and performs inference on test set

## Notes

- The configuration has been optimized for Kaggle's free tier with reduced model size and batch size
- Training is set to 5 epochs by default, which can be changed in cell2.md
- Checkpoints are saved in the `/kaggle/working/SegNext-med/checkpoints/` directory
- Visualizations are saved during training to help monitor progress

## Troubleshooting

If you encounter memory issues:
- Further reduce the batch size in cell2.md
- Reduce the model size by editing the `embed_dims` in cell6.md
- Reduce the image resolution in cell2.md 