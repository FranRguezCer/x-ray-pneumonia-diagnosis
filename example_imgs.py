import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Define paths
original_img_path = "data/chest_xray/train/NORMAL/IM-0115-0001.jpeg"  # Adjust the path if necessary
example_imgs_dir = "example_imgs"
os.makedirs(example_imgs_dir, exist_ok=True)
original_output_path = os.path.join(example_imgs_dir, "original_image.png")
preprocessed_output_path = os.path.join(example_imgs_dir, "preprocessed_image.png")
top_9_activations_path = os.path.join(example_imgs_dir, "top_9_activations.png")

# 1. Save Original Image
image = Image.open(original_img_path)
if image.mode != 'RGB':
    image = image.convert('RGB')
image.save(original_output_path)
print(f"Original image saved as {original_output_path}")

# 2. Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
preprocessed_image = preprocess(image)

# Convert back to PIL for visualization
preprocessed_image_pil = transforms.ToPILImage()(preprocessed_image)
preprocessed_image_pil.save(preprocessed_output_path)
print(f"Preprocessed image saved as {preprocessed_output_path}")

# 3. Generate Top 9 Convolutional Activations
model = models.resnet18(pretrained=True)
model.eval()

# Add batch dimension to the preprocessed image
preprocessed_image = preprocessed_image.unsqueeze(0)

# Get feature maps from the first convolutional layer
with torch.no_grad():
    feature_maps = model.conv1(preprocessed_image)

# Calculate variance for each feature map and select the top 9
variances = [feature_maps[0, i].var().item() for i in range(feature_maps.shape[1])]
top_indices = np.argsort(variances)[-9:]  # Indices of the 9 most interesting activations

# Plot and save the 9 most interesting activations in a 3x3 grid
plt.figure(figsize=(12, 12))
for i, idx in enumerate(top_indices):
    feature_map = feature_maps[0, idx].cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())  # Normalize for visualization
    plt.subplot(3, 3, i + 1)
    plt.imshow(feature_map, cmap='gray')
    plt.axis('off')
    plt.title(f"Activation {idx+1}")
plt.tight_layout()
plt.savefig(top_9_activations_path)
print(f"Grid of 9 most interesting activations saved as {top_9_activations_path}")
