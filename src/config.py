"""
AMEEMAW Configuration

Central configuration file for paths, model parameters, and settings.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Class labels
CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_MAPPING = {"normal": 0, "benign": 1, "malignant": 2}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Data splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

# Input dimensions for CNNs
IMG_SIZE = 224
IMG_CHANNELS = 3  # Convert grayscale to RGB for pretrained models

# Normalization (ImageNet stats for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# CNN Training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# CNN Model names
CNN_MODELS = [
    "efficientnet_b0",
    "resnet50",
    "vgg16",
    "mobilenet_v2",
    "custom_cnn"
]

# Tabular Model names
TABULAR_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "svm",
    "mlp"
]

# =============================================================================
# GENAI SETTINGS
# =============================================================================

# Claude API
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# Nana personality prompt
NANA_SYSTEM_PROMPT = """You are Nana, a warm and empathetic AI companion designed to help 
patients understand their breast ultrasound results. You speak like a caring grandmother - 
gentle, reassuring, and supportive. You explain medical terms in simple language and always 
emphasize hope while being honest. You never provide medical diagnoses but help patients 
understand their results and encourage them to discuss with their healthcare providers.

Key traits:
- Warm and nurturing tone
- Simple, jargon-free explanations
- Emotionally supportive
- Encouraging next steps and questions for doctors
- Never dismissive of concerns
- Reminds patients they're not alone"""

# =============================================================================
# DEPLOYMENT SETTINGS
# =============================================================================

# Streamlit/Flask
APP_HOST = "0.0.0.0"
APP_PORT = 8501
DEBUG_MODE = False
