# AMEEMAW Data Dictionary

## Overview

This document describes the data structures used in the AMEEMAW breast ultrasound classification project. It covers both the raw dataset (BUSI) and the processed/engineered features.

---

## Part 1: Raw Data (BUSI Dataset)

### Source Information
- **Dataset Name:** Breast Ultrasound Images Dataset (BUSI)
- **Source:** Kaggle (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Original Authors:** Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020)
- **License:** CC BY 4.0
- **Collection Site:** Baheya Hospital for Early Detection and Treatment of Women's Cancer, Cairo, Egypt

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Images | 780 |
| Image Format | PNG (Grayscale) |
| Average Dimensions | ~500×500 pixels |
| Classes | 3 (Normal, Benign, Malignant) |

### Raw Data Schema

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `image_path` | String | Relative path to ultrasound image file | `benign/benign (123).png` |
| `mask_path` | String (nullable) | Path to segmentation mask (if exists) | `benign/benign (123)_mask.png` |
| `class` | Categorical | Classification label | `normal`, `benign`, `malignant` |
| `filename` | String | Original image filename | `benign (123).png` |

### Class Distribution (Raw)

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 133 | 17.1% |
| Benign | 437 | 56.0% |
| Malignant | 210 | 26.9% |
| **Total** | **780** | **100%** |

### Image Properties (Raw)

| Property | Min | Max | Mean | Std |
|----------|-----|-----|------|-----|
| Width (pixels) | 300 | 1000 | ~500 | ~150 |
| Height (pixels) | 300 | 700 | ~500 | ~100 |
| Intensity (0-255) | 0 | 255 | ~95 | ~45 |

### Segmentation Masks
- **Format:** Binary PNG (0 = background, 255 = lesion)
- **Availability:** Available for Benign and Malignant classes only
- **Naming:** `{class} ({id})_mask.png`
- **Use:** Lesion size calculation, ROI extraction for radiomics

---

## Part 2: Processed Data (After Preprocessing)

### Train/Validation/Test Split

| Split | Count | Percentage | File |
|-------|-------|------------|------|
| Training | 546 | 70% | `data/processed/train.csv` |
| Validation | 117 | 15% | `data/processed/val.csv` |
| Test | 117 | 15% | `data/processed/test.csv` |

**Split Method:** Stratified random split preserving class proportions

### Processed Image Schema

| Field | Type | Description | Transformation |
|-------|------|-------------|----------------|
| `image_path` | String | Path to original image | Unchanged |
| `class` | Categorical | Classification label | Unchanged |
| `class_idx` | Integer | Numeric class label | 0=Normal, 1=Benign, 2=Malignant |
| `split` | Categorical | Dataset partition | `train`, `val`, `test` |

### Image Preprocessing Pipeline

| Step | Transformation | Parameters |
|------|---------------|------------|
| 1. Resize | Bilinear interpolation | 224×224 pixels |
| 2. Convert | Grayscale → RGB | 3 channels |
| 3. Normalize | ImageNet standardization | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| 4. Tensor | NumPy → PyTorch tensor | float32 |

### Data Augmentation (Training Only)

| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| Horizontal Flip | 50% | - |
| Rotation | 50% | ±15 degrees |
| Brightness | 30% | ±20% |
| Contrast | 30% | ±20% |

---

## Part 3: Radiomics Features (Feature Engineering)

### Overview
- **Library:** PyRadiomics v3.0+
- **Total Features:** 102 features extracted per image
- **Feature File:** `data/processed/radiomics_features.csv`
- **Normalization:** StandardScaler (mean=0, std=1)

### Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Shape (2D) | 10 | Geometric properties of lesion |
| First-Order | 18 | Intensity histogram statistics |
| GLCM | 24 | Gray Level Co-occurrence Matrix |
| GLRLM | 16 | Gray Level Run Length Matrix |
| GLSZM | 16 | Gray Level Size Zone Matrix |
| NGTDM | 5 | Neighborhood Gray Tone Difference |
| GLDM | 13 | Gray Level Dependence Matrix |
| **Total** | **102** | |

### Shape Features (2D) - 10 Features

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `shape2D_Sphericity` | Roundness (1=perfect circle) | Low → irregular shape → malignant |
| `shape2D_Elongation` | Length/width ratio | High → elongated → malignant |
| `shape2D_Perimeter` | Lesion boundary length | Large/irregular → malignant |
| `shape2D_MajorAxisLength` | Longest diameter | Tumor size indicator |
| `shape2D_MinorAxisLength` | Shortest diameter | Asymmetry indicator |
| `shape2D_MaximumDiameter` | Maximum Feret diameter | Overall lesion size |
| `shape2D_MeshSurface` | Surface area (2D) | Extent of lesion |
| `shape2D_PixelSurface` | Area in pixels | Lesion size |
| `shape2D_Perimeter/SurfaceRatio` | Boundary complexity | Irregular margins |
| `shape2D_Compactness` | Shape efficiency | Regular vs irregular |

### First-Order Statistics - 18 Features

| Feature | Description |
|---------|-------------|
| `firstorder_Mean` | Average intensity |
| `firstorder_Median` | Median intensity |
| `firstorder_Minimum` | Darkest pixel |
| `firstorder_Maximum` | Brightest pixel |
| `firstorder_Range` | Intensity range |
| `firstorder_Variance` | Intensity variation |
| `firstorder_StandardDeviation` | Intensity spread |
| `firstorder_Skewness` | Intensity asymmetry |
| `firstorder_Kurtosis` | Intensity tailedness |
| `firstorder_Energy` | Intensity magnitude |
| `firstorder_Entropy` | Intensity randomness |
| `firstorder_10Percentile` | 10th percentile intensity |
| `firstorder_90Percentile` | 90th percentile intensity |
| `firstorder_InterquartileRange` | IQR of intensity |
| `firstorder_MeanAbsoluteDeviation` | Average deviation |
| `firstorder_RobustMeanAbsoluteDeviation` | Robust MAD |
| `firstorder_RootMeanSquared` | RMS intensity |
| `firstorder_Uniformity` | Intensity uniformity |

### Texture Features - GLCM (24 Features)

Gray Level Co-occurrence Matrix features capture spatial relationships between pixels.

| Feature | Description |
|---------|-------------|
| `glcm_Autocorrelation` | Linear dependency |
| `glcm_ClusterProminence` | Asymmetry of GLCM |
| `glcm_ClusterShade` | Skewness of GLCM |
| `glcm_ClusterTendency` | Grouping of similar values |
| `glcm_Contrast` | Local intensity variation |
| `glcm_Correlation` | Linear dependency |
| `glcm_DifferenceAverage` | Average difference |
| `glcm_DifferenceEntropy` | Difference randomness |
| `glcm_DifferenceVariance` | Difference variation |
| `glcm_Id` | Inverse difference (homogeneity) |
| `glcm_Idm` | Inverse difference moment |
| `glcm_Idmn` | Normalized IDM |
| `glcm_Idn` | Normalized ID |
| `glcm_Imc1` | Information measure 1 |
| `glcm_Imc2` | Information measure 2 |
| `glcm_InverseVariance` | Inverse of variance |
| `glcm_JointAverage` | Mean of GLCM |
| `glcm_JointEnergy` | Angular second moment |
| `glcm_JointEntropy` | GLCM randomness |
| `glcm_MaximumProbability` | Most common pattern |
| `glcm_MCC` | Maximal correlation coefficient |
| `glcm_SumAverage` | Sum average |
| `glcm_SumEntropy` | Sum entropy |
| `glcm_SumSquares` | Sum of squares (variance) |

### Texture Features - GLRLM (16 Features)

Gray Level Run Length Matrix features capture texture coarseness.

| Feature | Description |
|---------|-------------|
| `glrlm_GrayLevelNonUniformity` | Variability of gray levels |
| `glrlm_GrayLevelNonUniformityNormalized` | Normalized GLNU |
| `glrlm_GrayLevelVariance` | Variance of gray levels |
| `glrlm_HighGrayLevelRunEmphasis` | Emphasis on bright runs |
| `glrlm_LongRunEmphasis` | Emphasis on long runs |
| `glrlm_LongRunHighGrayLevelEmphasis` | Long bright runs |
| `glrlm_LongRunLowGrayLevelEmphasis` | Long dark runs |
| `glrlm_LowGrayLevelRunEmphasis` | Emphasis on dark runs |
| `glrlm_RunEntropy` | Run randomness |
| `glrlm_RunLengthNonUniformity` | Run length variability |
| `glrlm_RunLengthNonUniformityNormalized` | Normalized RLNU |
| `glrlm_RunPercentage` | Run fraction |
| `glrlm_RunVariance` | Run length variance |
| `glrlm_ShortRunEmphasis` | Emphasis on short runs |
| `glrlm_ShortRunHighGrayLevelEmphasis` | Short bright runs |
| `glrlm_ShortRunLowGrayLevelEmphasis` | Short dark runs |

### Top 10 Most Important Features (from SHAP Analysis)

| Rank | Feature | Category | Importance | Direction |
|------|---------|----------|------------|-----------|
| 1 | `shape2D_Sphericity` | Shape | Highest | Low → Malignant |
| 2 | `shape2D_Elongation` | Shape | Very High | High → Malignant |
| 3 | `shape2D_Perimeter` | Shape | Very High | Large/Irregular → Malignant |
| 4 | `shape2D_MinorAxisLength` | Shape | High | Asymmetry indicator |
| 5 | `glrlm_RunEntropy` | Texture | High | High → Heterogeneous → Malignant |
| 6 | `shape2D_MajorAxisLength` | Shape | High | Size indicator |
| 7 | `firstorder_Entropy` | First-Order | Medium-High | High → Complex → Malignant |
| 8 | `glcm_Contrast` | Texture | Medium-High | High → Variable → Malignant |
| 9 | `shape2D_MaximumDiameter` | Shape | Medium | Size indicator |
| 10 | `glszm_ZoneEntropy` | Texture | Medium | Heterogeneity measure |

---

## Part 4: Bias Audit Features

### Brightness Bins

| Bin | Range (Mean Intensity) | Description |
|-----|----------------------|-------------|
| Dark | 0-80 | Low quality/dark images |
| Med-Dark | 81-95 | Below average brightness |
| Med-Light | 96-110 | Above average brightness |
| Light | 111-255 | High brightness images |

### Lesion Size Bins

| Bin | Range (Pixels) | Description |
|-----|----------------|-------------|
| Small | < 1000 | Small lesions (early stage) |
| Med-Small | 1000-5000 | Medium-small lesions |
| Med-Large | 5000-15000 | Medium-large lesions |
| Large | > 15000 | Large lesions |

---

## Part 5: Model Prediction Files

### Prediction Schema (`resnet50_predictions.csv`, `logreg_predictions.csv`)

| Field | Type | Description |
|-------|------|-------------|
| `image_path` | String | Path to test image |
| `true_label` | Integer | Ground truth class (0/1/2) |
| `pred_label` | Integer | Predicted class |
| `confidence` | Float | Prediction confidence (0-1) |
| `correct` | Boolean | Whether prediction was correct |
| `brightness_bin` | Categorical | Brightness category |
| `size_bin` | Categorical | Lesion size category |

---

## Notes

1. **Class Encoding:**
   - 0 = Normal
   - 1 = Benign  
   - 2 = Malignant

2. **Missing Values:**
   - No missing values in processed features
   - Masks only available for Benign/Malignant classes

3. **Feature Normalization:**
   - All radiomics features normalized using StandardScaler
   - Fit on training set, transform applied to val/test

4. **Reproducibility:**
   - Random seed: 42
   - All preprocessing parameters documented in notebooks

---

*Last Updated: December 2025*
*Author: Shira Amina Ortiz Olivares*
