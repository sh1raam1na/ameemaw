"""
Model utilities for AMEEMAW.
Handles ResNet-50 model loading and breast ultrasound classification.

ALIGNED WITH: 03_model_training.ipynb
Architecture: ResNet-50 with Dropout(0.3) + Linear(2048, 3)
Class Encoding: Normal=0, Benign=1, Malignant=2
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Optional
import os


# ============================================
# CONSTANTS (from 03_model_training.ipynb)
# ============================================

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class labels (from 02_feature_engineering.ipynb)
# Encoding: normal=0, benign=1, malignant=2
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
CLASS_ENCODING = {'normal': 0, 'benign': 1, 'malignant': 2}

# Model metrics (from 05_bias_audit.ipynb & 06_genai_integration.ipynb)
MODEL_METRICS = {
    'overall_accuracy': 0.846,
    'malignant_recall': 0.839,
    'malignant_precision': 0.722,
    'benign_recall': 0.818,
    'normal_recall': 0.950,
    'small_malignant_recall': 0.50,  # Critical limitation!
    'large_lesion_accuracy': 0.708,
    'high_confidence_threshold': 0.90,
    'medium_confidence_threshold': 0.70,
}

# BI-RADS mapping and information
CLASS_INFO = {
    'Normal': {
        'birads': '1',
        'description': 'No suspicious findings detected. The breast tissue appears typical with no visible lumps or concerning areas.',
        'characteristics': [
            'Homogeneous echotexture',
            'No focal masses',
            'Normal tissue architecture',
            'Well-defined layers'
        ],
        'recommendation': 'Routine screening as recommended by your healthcare provider.',
        'color': '#B5C4B1'  # Gentle sage
    },
    'Benign': {
        'birads': '2-3',
        'description': 'Benign (non-cancerous) characteristics observed. A mass is present but shows features typical of benign growths.',
        'characteristics': [
            'Oval or round shape',
            'Circumscribed (well-defined) margins',
            'Parallel orientation to skin',
            'Uniform internal echoes',
            'Possible simple cyst features'
        ],
        'recommendation': 'May require short-term follow-up. Consult with your healthcare provider.',
        'color': '#F7E1AE'  # Sunshine yellow
    },
    'Malignant': {
        'birads': '4-5',
        'description': 'Suspicious features that warrant further evaluation. The AI detected features it associates with potentially concerning masses.',
        'characteristics': [
            'Irregular shape',
            'Spiculated or indistinct margins',
            'Non-parallel (vertical) orientation',
            'Heterogeneous internal echoes',
            'Posterior acoustic shadowing',
            'Possible microcalcifications'
        ],
        'recommendation': 'Biopsy recommended. Please consult with your healthcare provider promptly.',
        'color': '#E8B4B8'  # Primary rose
    }
}


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_transforms() -> transforms.Compose:
    """
    Get image preprocessing transforms for inference.
    MUST match eval_transforms from 03_model_training.ipynb!
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_resnet50(num_classes: int = 3) -> nn.Module:
    """
    Create ResNet-50 model with architecture matching training.
    
    FROM 03_model_training.ipynb:
    ```
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    ```
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


class BreastUltrasoundClassifier(nn.Module):
    """
    Wrapper for ResNet-50 classifier matching 03_model_training.ipynb architecture.
    """
    
    def __init__(self, num_classes: int = 3):
        super(BreastUltrasoundClassifier, self).__init__()
        self.backbone = get_resnet50(num_classes)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @property
    def layer4(self):
        """Access layer4 for Grad-CAM."""
        return self.backbone.layer4
    
    @property
    def fc(self):
        """Access fc layer."""
        return self.backbone.fc


def load_model(model_path: Optional[str] = None) -> nn.Module:
    """
    Load the trained ResNet-50 model.
    
    Args:
        model_path: Path to saved model weights (.pth file)
        
    Returns:
        Loaded model ready for inference
    """
    device = get_device()
    
    # Create model with same architecture as training
    model = get_resnet50(num_classes=3)
    
    # Search for model weights
    if model_path and os.path.exists(model_path):
        weights_path = model_path
    else:
        # Default paths to search (matching your repo structure)
        default_paths = [
            '../models/resnet_50_best.pth',      # From app/ folder
            'models/resnet_50_best.pth',          # From root
            '../models/resnet50_best.pth',
            'models/resnet50_best.pth',
            'resnet_50_best.pth',
        ]
        
        weights_path = None
        for path in default_paths:
            if os.path.exists(path):
                weights_path = path
                break
    
    # Load weights if found
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from {weights_path}")
        except Exception as e:
            print(f"⚠️ Could not load weights from {weights_path}: {e}")
            print("   Using pretrained ImageNet weights (demo mode)")
    else:
        print("⚠️ No trained weights found. Using pretrained ImageNet weights.")
        print("   For production, add your resnet50_best.pth to models/ folder")
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for model input.
    Matches eval_transforms from training.
    
    Args:
        image: PIL Image (will be converted to RGB if needed)
        
    Returns:
        Preprocessed tensor ready for model input (1, 3, 224, 224)
    """
    # Ensure RGB (training used .convert('RGB'))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = get_transforms()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor


def predict_class(
    model: nn.Module,
    image: Image.Image
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict the class of a breast ultrasound image.
    
    Args:
        model: The trained ResNet-50 model
        image: PIL Image (RGB or grayscale)
        
    Returns:
        Tuple of (predicted_class_name, confidence, probabilities_dict)
    """
    device = get_device()
    
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    
    # Convert to Python types
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_value = confidence.item()
    
    # Create probability dictionary
    probs_dict = {
        CLASS_NAMES[i]: probabilities[0][i].item()
        for i in range(len(CLASS_NAMES))
    }
    
    return predicted_class, confidence_value, probs_dict


def get_confidence_level(confidence: float) -> str:
    """
    Categorize confidence level based on thresholds from 06_genai_integration.ipynb
    """
    if confidence >= MODEL_METRICS['high_confidence_threshold']:
        return 'high'
    elif confidence >= MODEL_METRICS['medium_confidence_threshold']:
        return 'medium'
    else:
        return 'low'


def get_safety_flags(
    predicted_class: str,
    confidence: float
) -> list:
    """
    Generate safety flags based on prediction.
    From 06_genai_integration.ipynb safety logic.
    """
    flags = []
    
    # Always warn about model limitations for benign predictions
    if predicted_class == 'Benign':
        miss_rate = 1 - MODEL_METRICS['malignant_recall']
        flags.append({
            'type': 'benign_caution',
            'severity': 'high',
            'message': f"Our model misses ~{miss_rate:.0%} of malignant cases. A 'Benign' prediction should NEVER replace professional evaluation."
        })
    
    # Warn about small lesion limitation
    flags.append({
        'type': 'small_lesion_warning',
        'severity': 'medium',
        'message': f"Our model only catches {MODEL_METRICS['small_malignant_recall']:.0%} of small cancers."
    })
    
    # Low confidence warning
    if confidence < MODEL_METRICS['medium_confidence_threshold']:
        flags.append({
            'type': 'low_confidence',
            'severity': 'medium',
            'message': f"Low confidence prediction ({confidence:.0%}). Results are uncertain."
        })
    
    return flags


def get_class_info(class_name: str) -> Dict:
    """
    Get detailed information about a classification.
    
    Args:
        class_name: One of 'Normal', 'Benign', 'Malignant'
        
    Returns:
        Dictionary containing class information including BI-RADS
    """
    return CLASS_INFO.get(class_name, CLASS_INFO['Normal'])


def get_prediction_context(
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float]
) -> Dict:
    """
    Build full context dictionary for Nana's response.
    Matches format expected by 06_genai_integration.ipynb
    """
    return {
        'mode': 'learn',
        'prediction': predicted_class,
        'confidence': confidence,
        'confidence_level': get_confidence_level(confidence),
        'probabilities': probabilities,
        'safety_flags': get_safety_flags(predicted_class, confidence),
        'class_info': get_class_info(predicted_class),
        'model_metrics': MODEL_METRICS
    }


# Demo/test function
def demo_prediction(image_path: str = None):
    """Demo function to test the model."""
    print("\n" + "="*50)
    print("🐧 AMEEMAW Model Demo")
    print("="*50)
    
    # Create dummy image if no path
    if image_path is None:
        dummy_image = Image.fromarray(
            np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        )
        image = dummy_image
        print("Using random dummy image")
    else:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded: {image_path}")
    
    # Load model
    model = load_model()
    
    # Predict
    predicted_class, confidence, probabilities = predict_class(model, image)
    
    # Display results
    print(f"\n📊 Prediction: {predicted_class}")
    print(f"   Confidence: {confidence:.1%} ({get_confidence_level(confidence)})")
    print(f"\n   Probabilities:")
    for cls, prob in probabilities.items():
        print(f"     {cls}: {prob:.1%}")
    
    # Safety flags
    flags = get_safety_flags(predicted_class, confidence)
    if flags:
        print(f"\n⚠️ Safety Flags:")
        for flag in flags:
            print(f"   [{flag['severity'].upper()}] {flag['message']}")
    
    print("="*50)
    
    return predicted_class, confidence, probabilities


if __name__ == "__main__":
    demo_prediction()
