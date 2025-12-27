"""
Grad-CAM utilities for AMEEMAW.
Provides visual explanations of model predictions.

ALIGNED WITH: 04_explainability.ipynb
Target Layer: model.layer4[-1] (last block of ResNet-50's layer4)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import torchvision.transforms as transforms


# ============================================
# CONSTANTS (matching 03_model_training.ipynb)
# ============================================

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_transforms() -> transforms.Compose:
    """Get eval transforms matching training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Visualizes which regions of an image are important for a
    CNN's prediction by using gradients flowing into the final
    convolutional layer.
    
    Reference: Selvaraju et al., ICCV 2017
    
    ALIGNED WITH: 04_explainability.ipynb implementation
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model (ResNet-50)
            target_layer: Layer to extract activations from
                         For ResNet-50: model.layer4[-1]
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input (1, 3, 224, 224)
            target_class: Class index to visualize (None = predicted class)
        
        Returns:
            cam: Heatmap array (H, W) normalized to [0, 1]
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        if target_class is None:
            target_class = pred_class
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep only positive influences
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, pred_class, confidence


def overlay_gradcam(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    ALIGNED WITH: 04_explainability.ipynb overlay_gradcam function
    
    Args:
        image: Original image as numpy array (H, W, 3) uint8
               OR PIL Image
        cam: Grad-CAM heatmap (h, w) normalized to [0, 1]
        alpha: Blend factor for overlay (default 0.5)
        colormap: OpenCV colormap (default JET)
    
    Returns:
        overlay: Blended image as PIL Image
        cam_resized: Resized heatmap matching image dimensions
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = np.uint8(alpha * cam_colored + (1 - alpha) * image)
    
    return Image.fromarray(overlay), cam_resized


def generate_gradcam(
    model,
    image: Image.Image,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for an image.
    
    Convenience function that handles preprocessing.
    
    Args:
        model: The trained ResNet-50 model
        image: PIL Image (RGB)
        target_class: Class to visualize (None = use predicted)
    
    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1]
        Resized to match original image dimensions
    """
    device = get_device()
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get target layer (layer4[-1] for ResNet-50)
    # Handle both direct model and wrapper
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    elif hasattr(model, 'backbone'):
        target_layer = model.backbone.layer4[-1]
    else:
        raise ValueError("Cannot find layer4 in model")
    
    # Create Grad-CAM and generate
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class, confidence = grad_cam.generate(input_tensor, target_class)
    
    # Resize to original image size
    original_size = image.size  # (W, H)
    cam_resized = cv2.resize(cam, original_size)
    
    return cam_resized


def get_attention_regions(
    heatmap: np.ndarray,
    threshold: float = 0.5
) -> list:
    """
    Extract high-attention regions from heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
        threshold: Minimum attention value to consider
    
    Returns:
        List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    # Threshold the heatmap
    binary = (heatmap > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter small regions
            boxes.append((x, y, x + w, y + h))
    
    return boxes


def create_explanation_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    probabilities: dict
) -> Image.Image:
    """
    Create a comprehensive explanation visualization.
    
    Args:
        image: Original PIL Image
        heatmap: Grad-CAM heatmap
        prediction: Predicted class name
        confidence: Prediction confidence
        probabilities: Dict of class probabilities
    
    Returns:
        Combined visualization as PIL Image
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Ultrasound', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlay, _ = overlay_gradcam(image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {prediction}\nConfidence: {confidence:.1%}', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf).copy()
    plt.close()
    
    return result


# Demo function
def demo_gradcam():
    """Demo Grad-CAM generation."""
    from utils.model import load_model
    
    print("\n" + "="*50)
    print("🔍 Grad-CAM Demo")
    print("="*50)
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    )
    
    # Load model
    model = load_model()
    
    # Generate Grad-CAM
    heatmap = generate_gradcam(model, dummy_image)
    overlay, _ = overlay_gradcam(dummy_image, heatmap)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print("✅ Grad-CAM generation successful!")
    
    return overlay


if __name__ == "__main__":
    demo_gradcam()
