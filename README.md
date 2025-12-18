# AMEEMAW 🩺💜

**AI Mamma-Echography Educator & Empathetic Mentor for Awareness & Wellness**

A breast ultrasound classification tool with explainable AI and GenAI-powered patient communication.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

AMEEMAW is an educational and supportive AI tool designed to:
1. Classify breast ultrasound images (Normal / Benign / Malignant)
2. Provide explainable predictions using Grad-CAM and SHAP
3. Offer empathetic, patient-friendly explanations through **Nana**, an AI companion

### Two Modes

| Mode | Purpose | Features |
|------|---------|----------|
| **Learn Mode** | Educational | Upload image → Guess → AI reveals prediction + Grad-CAM + explanation |
| **Explain Mode** | Emotional Support | Input BI-RADS result → Nana explains in plain language + next steps |

---

## 📊 Dataset

**Breast Ultrasound Images Dataset (BUSI)**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Size**: 780 images
- **Classes**: Normal (133), Benign (437), Malignant (210)
- **Format**: PNG grayscale images with corresponding masks

---

## 🏗️ Project Structure

```
ameemaw/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                    # Original BUSI dataset (gitignored)
│   ├── processed/              # Preprocessed images
│   └── data_dictionary.md      # Feature descriptions
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability.ipynb
│   ├── 05_bias_audit.ipynb
│   └── 06_genai_integration.ipynb
│
├── src/
│   ├── data/                   # Data loading & preprocessing
│   ├── features/               # Radiomics feature extraction
│   ├── models/                 # CNN & tabular model definitions
│   ├── explainability/         # Grad-CAM & SHAP implementations
│   ├── genai/                  # Claude API integration
│   └── utils/                  # Helper functions
│
├── app/                        # Streamlit/Flask web application
├── models/                     # Saved model weights
├── reports/                    # Generated analysis reports
├── presentations/              # Technical & business presentations
└── docs/                       # Setup & deployment guides
```

---

## 🤖 Models

### CNN Models (Image Classification)
- EfficientNet-B0
- ResNet-50
- VGG-16
- MobileNetV2
- Custom CNN

### Tabular Models (Radiomics Features)
- Logistic Regression
- Random Forest
- XGBoost
- SVM (RBF kernel)
- MLP

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ameemaw.git
cd ameemaw
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and extract to `data/raw/`

### 5. Run the notebooks
Start with `notebooks/01_eda.ipynb` and proceed sequentially.

### 6. Launch the app
```bash
streamlit run app/app.py
```

---

## 📈 Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| EfficientNet-B0 | TBD | TBD | TBD |
| ResNet-50 | TBD | TBD | TBD |
| ... | ... | ... | ... |

*Results will be updated after model training.*

---

## 🔍 Explainability

- **Grad-CAM**: Visual heatmaps showing which regions influenced CNN predictions
- **SHAP**: Feature importance analysis for tabular models
- **Bias Audit**: Analysis across age groups, breast density, image quality, and lesion size

---

## 💜 Meet Nana

Nana is AMEEMAW's AI companion — warm, empathetic, and supportive like a caring grandmother. She helps patients understand their results without medical jargon and provides emotional support during a stressful time.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BUSI Dataset: Al-Dhabyani et al. (2020)
- Anthropic Claude API for GenAI capabilities
- Course instructors and mentors

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
