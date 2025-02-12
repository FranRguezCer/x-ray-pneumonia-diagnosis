import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from typing import Tuple
import numpy as np

def get_data_loaders(data_dir: str, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for training, validation, and test datasets with better augmentation and split.
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),  # Increased rotation range
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Normalization only for validation/test (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset and split into train/validation
    full_train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    
    # Split dataset: 85% for training, 15% for validation
    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Calculate class weights for train_dataset
    class_counts = np.bincount([full_train_dataset.targets[i] for i in train_dataset.indices])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[full_train_dataset.targets[i]] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
