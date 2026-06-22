"""
src/evaluate.py
=================
Evaluation utilities: confusion matrix, classification report, sample predictions,
training curve plots.

Usage:
    python src/evaluate.py --dataset emnist_balanced
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config
from src.dataset import get_dataloaders
from src.model import CharCNN


def load_model(dataset_key: str, device):
    ckpt_path = config.checkpoint_path(dataset_key)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train the model first with:\n"
            f"  python src/train.py --dataset {dataset_key}"
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CharCNN(num_classes=checkpoint["num_classes"], in_channels=checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["class_names"]


@torch.no_grad()
def get_all_predictions(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    n = len(class_names)
    fig_size = max(8, n * 0.35)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=n <= 20, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix {'(normalized)' if normalize else ''}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history_path, save_path):
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}, skipping training curve plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["val_loss"], label="Val Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(history["val_acc"], label="Val Acc", marker='o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_sample_predictions(model, loader, class_names, device, save_path, n_samples=16):
    x, y = next(iter(loader))
    x_dev = x.to(device)
    with torch.no_grad():
        logits = model(x_dev)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu()
        confidences = probs.max(dim=1)[0].cpu()

    n_samples = min(n_samples, x.size(0))
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    # Un-normalize for display
    mean, std = config.NORMALIZE_MEAN, config.NORMALIZE_STD

    for i in range(n_samples):
        img = x[i, 0].numpy() * std + mean
        true_label = class_names[y[i].item()]
        pred_label = class_names[preds[i].item()]
        conf = confidences[i].item()
        correct = (y[i].item() == preds[i].item())

        axes[i].imshow(img, cmap="gray")
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {true_label} | Pred: {pred_label}\n({conf:.1%})",
                           color=color, fontsize=9)
        axes[i].axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model")
    parser.add_argument("--dataset", type=str, default=config.DATASET,
                         choices=list(config.DATASET_CHOICES.keys()))
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Using device: {device}")

    model, class_names = load_model(args.dataset, device)
    print(f"Loaded model for dataset: {args.dataset} ({len(class_names)} classes)")

    _, _, test_loader, num_classes, _ = get_dataloaders(
        dataset_key=args.dataset, augment=False
    )

    print("Running inference on test set...")
    preds, labels, probs = get_all_predictions(model, test_loader, device)

    test_acc = (preds == labels).mean()
    print(f"\nOverall Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    report = classification_report(labels, preds, target_names=[str(c) for c in class_names],
                                    zero_division=0)
    print(report)

    report_path = os.path.join(config.OUTPUT_DIR, f"classification_report_{args.dataset}.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Visualizations
    cm_path = os.path.join(config.OUTPUT_DIR, f"confusion_matrix_{args.dataset}.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path, normalize=True)

    curves_path = os.path.join(config.OUTPUT_DIR, f"training_curves_{args.dataset}.png")
    plot_training_curves(config.history_path(args.dataset), curves_path)

    samples_path = os.path.join(config.OUTPUT_DIR, f"sample_predictions_{args.dataset}.png")
    plot_sample_predictions(model, test_loader, class_names, device, samples_path)


if __name__ == "__main__":
    main()
