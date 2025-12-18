# AMEEMAW Data Dictionary

## Dataset: Breast Ultrasound Images (BUSI)

### Source
- **Origin**: Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020)
- **Paper**: "Dataset of Breast Ultrasound Images"
- **Kaggle**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

---

## Raw Data Structure

```
Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png          # Ultrasound image
│   ├── benign (1)_mask.png     # Segmentation mask
│   ├── benign (1)_mask_1.png   # Additional mask (if multiple lesions)
│   └── ...
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   └── ...
└── normal/
    ├── normal (1).png
    └── ...                      # No masks for normal images
```

---

## Class Distribution

| Class | Count | Percentage | Has Mask |
|-------|-------|------------|----------|
| Normal | 133 | 17.1% | No |
| Benign | 437 | 56.0% | Yes |
| Malignant | 210 | 26.9% | Yes |
| **Total** | **780** | **100%** | - |

---

## Image Specifications

| Attribute | Value |
|-----------|-------|
| Format | PNG |
| Color Mode | Grayscale (converted from RGB) |
| Resolution | Variable (avg ~500x500) |
| Bit Depth | 8-bit |

---

## Feature Definitions

### Raw Image Features (After Preprocessing)

| Feature | Type | Description |
|---------|------|-------------|
| `image_path` | string | Path to original image file |
| `mask_path` | string | Path to segmentation mask (if exists) |
| `class` | categorical | Target label: normal, benign, malignant |
| `class_encoded` | int | Encoded label: 0=normal, 1=benign, 2=malignant |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `has_mask` | bool | Whether segmentation mask exists |

### Extracted Radiomics Features

| Feature Group | Count | Description |
|---------------|-------|-------------|
| **First Order** | ~18 | Statistical features (mean, std, skewness, etc.) |
| **Shape** | ~14 | Geometric properties (area, perimeter, sphericity) |
| **GLCM** | ~24 | Gray Level Co-occurrence Matrix features |
| **GLRLM** | ~16 | Gray Level Run Length Matrix features |
| **GLSZM** | ~16 | Gray Level Size Zone Matrix features |
| **GLDM** | ~14 | Gray Level Dependence Matrix features |
| **NGTDM** | ~5 | Neighboring Gray Tone Difference Matrix |

### Key Radiomics Features (Subset)

| Feature | Type | Description |
|---------|------|-------------|
| `mean_intensity` | float | Average pixel intensity |
| `std_intensity` | float | Standard deviation of pixel intensity |
| `skewness` | float | Asymmetry of intensity distribution |
| `kurtosis` | float | Tailedness of intensity distribution |
| `entropy` | float | Randomness/complexity of texture |
| `contrast` | float | Local intensity variation |
| `homogeneity` | float | Closeness of element distribution |
| `energy` | float | Sum of squared elements in GLCM |
| `correlation` | float | Linear dependency of gray levels |
| `area` | float | Lesion area (from mask) |
| `perimeter` | float | Lesion boundary length |
| `circularity` | float | How circular the lesion is |
| `eccentricity` | float | Elongation of the lesion |

---

## Data Splits

| Split | Percentage | Stratified |
|-------|------------|------------|
| Train | 70% | Yes |
| Validation | 15% | Yes |
| Test | 15% | Yes |

---

## Preprocessing Pipeline

1. **Load**: Read PNG images as grayscale
2. **Resize**: Standardize to 224x224 (for CNN input)
3. **Normalize**: Scale pixel values to [0, 1]
4. **Augmentation** (train only):
   - Random horizontal flip
   - Random rotation (±15°)
   - Random brightness/contrast adjustment

---

## Quality Notes

- Some images have multiple masks (multiple lesions) - we use the primary mask
- Normal class has no masks - excluded from mask-based radiomics
- Image quality varies - some images have artifacts or annotations

---

## Ethical Considerations

- Data is de-identified (no patient information)
- Used for educational purposes only
- Not suitable for clinical diagnosis
- Bias analysis required across demographic proxies
