import csv
import json
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union
from sklearn.metrics import classification_report

def ensure_output_dir(output_dir: str = "output"):
    """
    Ensures that the output directory exists. If not, it creates it.
    
    Args:
        output_dir (str): Path to the output directory. Default is "output".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def log_metrics(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float, log_file: str = "output/metrics.csv"):
    """
    Logs training and validation metrics to a CSV file, overwriting the file at the start of each execution.
    
    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        val_loss (float): Validation loss for the epoch.
        val_acc (float): Validation accuracy for the epoch.
        log_file (str): Path to the CSV file where metrics are logged. Default is "output/metrics.csv".
    """
    ensure_output_dir(os.path.dirname(log_file))
    
    mode = 'w' if epoch == 1 else 'a'
    with open(log_file, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

def plot_metrics(log_file: str = "output/metrics.csv", output_file: str = "output/performance_plot.png", model_name: str = "ResNet18"):
    """
    Plots the training and validation metrics from the log file and saves the plot as an image.
    
    Args:
        log_file (str): Path to the CSV file containing the logged metrics.
        output_file (str): Path to save the output plot image.
        model_name (str): Name of the model to include in the plot title. Default is "ResNet18".
    """
    metrics = pd.read_csv(log_file)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.figure(figsize=(14, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics["Epoch"], metrics["Train Loss"], label="Train Loss", color="blue", alpha=0.7)
    plt.fill_between(metrics["Epoch"], metrics["Train Loss"], alpha=0.2, color="blue")
    plt.plot(metrics["Epoch"], metrics["Val Loss"], label="Val Loss", color="orange", alpha=0.7)
    plt.fill_between(metrics["Epoch"], metrics["Val Loss"], alpha=0.2, color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics["Epoch"], metrics["Train Accuracy"], label="Train Accuracy", color="green", alpha=0.7)
    plt.fill_between(metrics["Epoch"], metrics["Train Accuracy"], alpha=0.2, color="green")
    plt.plot(metrics["Epoch"], metrics["Val Accuracy"], label="Val Accuracy", color="red", alpha=0.7)
    plt.fill_between(metrics["Epoch"], metrics["Val Accuracy"], alpha=0.2, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.suptitle(f"Model: {model_name} | Date: {current_date}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ensure_output_dir(os.path.dirname(output_file))
    plt.savefig(output_file)
    print(f"Performance plot saved as {output_file}")

def save_model(model: torch.nn.Module, path: str = "output/radiography_model.pth"):
    """
    Saves the trained model to a file.
    
    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the model. Default is "output/radiography_model.pth".
    """
    ensure_output_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def generate_report_json(all_labels: list, all_preds: list, output_file: str = "output/report.json"):
    """
    Generates a classification report in JSON format.
    
    Args:
        all_labels (list): True labels from the test set.
        all_preds (list): Predicted labels from the model.
        output_file (str): Path to save the JSON report. Default is "output/report.json".
    """
    ensure_output_dir(os.path.dirname(output_file))
    report = classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved as {output_file}")

def generate_report_markdown(all_labels: list, all_preds: list, output_file: str = "output/report.md"):
    """
    Generates a classification report in Markdown table format.
    
    Args:
        all_labels (list): True labels from the test set.
        all_preds (list): Predicted labels from the model.
        output_file (str): Path to save the Markdown report. Default is "output/report.md".
    """
    ensure_output_dir(os.path.dirname(output_file))
    report = classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    with open(output_file, 'w') as f:
        f.write("# Classification Report\n\n")
        f.write(report_df.to_markdown(index=True))
    print(f"Classification report saved as {output_file}")