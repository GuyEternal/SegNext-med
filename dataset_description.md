# ISIC-2017 Skin Lesion Dataset

## Overview
The ISIC-2017 dataset is a comprehensive collection of dermoscopic images of skin lesions, designed for developing automated skin cancer classification and segmentation algorithms. The dataset was released as part of the International Skin Imaging Collaboration (ISIC) 2017 challenge, focusing on skin lesion analysis for melanoma detection.

## Dataset Structure
The dataset is organized into three distinct subsets:
- **Training Set**: 2000 dermoscopic images with corresponding ground truth
- **Validation Set**: 150 images with ground truth
- **Test Set**: 600 images with ground truth

Each subset contains:
1. **Dermoscopic Images**: High-resolution clinical images (JPG format)
2. **Superpixel Images**: Pre-processed versions of the originals for advanced analysis

## Ground Truth Data
The ground truth data is divided into three parts:

### Part 1: Lesion Segmentation Masks
- Binary PNG images corresponding to each dermoscopic image
- White pixels represent the lesion area, black pixels represent the background
- Used for lesion segmentation tasks
- Filename format: `ISIC_XXXXXXX_segmentation.png`

### Part 2: Dermoscopic Feature Extraction
- JSON files containing binary masks for five dermoscopic features:
  - Pigment Network
  - Negative Network
  - Milia-like Cysts
  - Streaks
  - Other features
- Each feature is represented as a binary array
- These features are clinically significant indicators used by dermatologists
- Filename format: `ISIC_XXXXXXX_features.json`

### Part 3: Lesion Classification
- CSV files containing diagnosis labels for each image:
  - melanoma (binary: 0 or 1)
  - seborrheic_keratosis (binary: 0 or 1)
  - If both labels are 0, the lesion is considered benign
- Filename format: `ISIC-2017_[Subset]_Part3_GroundTruth.csv`

## Dataset Statistics
- **Image Format**: High-resolution JPG (dermoscopic images)
- **Mask Format**: PNG (segmentation ground truth)
- **Feature Format**: JSON (dermoscopic features)
- **Label Format**: CSV (diagnosis)
- **Class Distribution**: Imbalanced, with a minority of melanoma and seborrheic keratosis cases
- **Resolution**: Variable, with most images being several megabytes in size

## Clinical Significance
- Melanoma is the deadliest form of skin cancer
- Early detection is critical for successful treatment
- Automated analysis can assist dermatologists in screening and diagnosis
- The dermoscopic features (Part 2) are key visual indicators used by specialists

## Technical Considerations
- Images contain substantial variation in:
  - Lighting conditions
  - Skin types
  - Lesion sizes
  - Presence of artifacts (hair, ruler markings, air bubbles)
- The dataset is suitable for three tasks:
  1. Lesion segmentation (Part 1)
  2. Feature extraction (Part 2)
  3. Disease classification (Part 3)

## Directory Structure
```
isic-2017-kaggle/
├── ISIC-2017_Training_Data/
│   └── ISIC-2017_Training_Data/ (dermoscopic images)
├── ISIC-2017_Training_Part1_GroundTruth/
│   └── ISIC-2017_Training_Part1_GroundTruth/ (segmentation masks)
├── ISIC-2017_Training_Part2_GroundTruth/
│   └── ISIC-2017_Training_Part2_GroundTruth/ (dermoscopic features)
├── ISIC-2017_Training_Part3_GroundTruth.csv (diagnosis labels)
├── ISIC-2017_Validation_Data/
│   └── ISIC-2017_Validation_Data/ (dermoscopic images)
├── ISIC-2017_Validation_Part1_GroundTruth/
│   └── ISIC-2017_Validation_Part1_GroundTruth/ (segmentation masks)
├── ISIC-2017_Validation_Part2_GroundTruth/
│   └── ISIC-2017_Validation_Part2_GroundTruth/ (dermoscopic features)
├── ISIC-2017_Validation_Part3_GroundTruth.csv (diagnosis labels)
├── ISIC-2017_Test_v2_Data/
│   └── ISIC-2017_Test_v2_Data/ (dermoscopic images)
├── ISIC-2017_Test_v2_Part1_GroundTruth/
│   └── ISIC-2017_Test_v2_Part1_GroundTruth/ (segmentation masks)
├── ISIC-2017_Test_v2_Part2_GroundTruth/
│   └── ISIC-2017_Test_v2_Part2_GroundTruth/ (dermoscopic features)
└── ISIC-2017_Test_v2_Part3_GroundTruth.csv (diagnosis labels)
```

## Citation
The dataset was created by the International Skin Imaging Collaboration (ISIC) and released as part of the ISIC 2017 Challenge. Proper attribution should be given when using this dataset for research purposes.

## Potential Uses with LLMs
This dataset can be used with Large Language Models for several purposes:
1. **Medical image analysis**: Training models to analyze dermatological images
2. **Multimodal learning**: Combining image analysis with textual descriptions
3. **Report generation**: Automatic generation of diagnostic reports
4. **Feature extraction guidance**: Teaching LLMs to identify and describe clinical features
5. **Patient education**: Creating explanatory content for patients based on image analysis
6. **Clinical decision support**: Assisting healthcare providers in diagnosis and treatment planning

When using with LLMs, consider converting images to text-based features or embedding representations that can be processed alongside the textual components of the dataset. 