# AMEEMAW Setup Guide

## Prerequisites

- Python 3.9 or higher
- Git
- ~4GB disk space (for data + models)
- GPU recommended for CNN training (or use Google Colab)

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ameemaw.git
cd ameemaw
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n ameemaw python=3.9
conda activate ameemaw
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for PyTorch with GPU:**
If you have a CUDA-compatible GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download the Dataset

1. Go to [BUSI Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
2. Download and extract to `data/raw/`
3. Your structure should look like:
   ```
   data/raw/
   └── Dataset_BUSI_with_GT/
       ├── benign/
       ├── malignant/
       └── normal/
   ```

### 5. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key (for Nana companion).

### 6. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "from src.config import PROJECT_ROOT; print(f'Project root: {PROJECT_ROOT}')"
```

---

## Google Colab Setup

For GPU-accelerated training without local GPU:

### 1. Open Colab Notebook

Upload notebooks to Google Colab or mount from Google Drive.

### 2. Mount Drive (if using Drive)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone Repo in Colab

```python
!git clone https://github.com/YOUR_USERNAME/ameemaw.git
%cd ameemaw
```

### 4. Install Dependencies

```python
!pip install -r requirements.txt
```

### 5. Upload Dataset

Upload the BUSI dataset to your Google Drive and copy:
```python
!cp -r /content/drive/MyDrive/Dataset_BUSI_with_GT data/raw/
```

---

## Troubleshooting

### PyTorch CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Pyradiomics Installation Issues
```bash
# Try installing with specific version
pip install pyradiomics==3.0.1
```

### Memory Issues
- Reduce `BATCH_SIZE` in `src/config.py`
- Use gradient checkpointing
- Switch to smaller models (MobileNetV2)

---

## Next Steps

After setup, start with the notebooks in order:
1. `01_eda.ipynb` - Explore the data
2. `02_feature_engineering.ipynb` - Prepare features
3. Continue sequentially...

See [README.md](../README.md) for full project overview.
