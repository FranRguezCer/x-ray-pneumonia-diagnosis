import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def build_model() -> nn.Module:
    """
    Builds a ResNet18 model pre-trained on ImageNet and modifies the final layer for binary classification.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),  # Dropout to reduce overfitting
        nn.Linear(num_features, 2)
    )
    
    return model
