import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_data_loaders
from src.model import build_model
from src.utils import log_metrics, plot_metrics, save_model, generate_report_json, generate_report_markdown
from sklearn.metrics import classification_report

# Detect if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset with batch_size=32
data_dir = 'data/chest_xray'
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=16)

# Build model
model = build_model().to(device)
criterion = nn.CrossEntropyLoss()

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.7)

# Number of training epochs
num_epochs = 15
best_val_loss = float('inf')
patience_counter = 0
patience = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
    
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
    
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    
    # Log training and validation metrics
    log_metrics(epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model(model, path="output/best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Generate performance plot
plot_metrics(model_name="ResNet18")

# Test phase
print("\n--- Evaluating on Test Set ---")
model.eval()
test_loss = 0.0
test_accuracy = 0.0
all_labels, all_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())

test_loss /= len(test_loader)
test_accuracy /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

# Generate and save classification report as JSON and Markdown
generate_report_json(all_labels, all_preds)
generate_report_markdown(all_labels, all_preds)