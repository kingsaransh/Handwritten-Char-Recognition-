import sys
import os
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from src.dataset import get_dataloaders
from src.model import CharCNN


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train CNN for handwritten character recognition")
    parser.add_argument("--dataset", type=str, default=config.DATASET,
                         choices=list(config.DATASET_CHOICES.keys()))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        dataset_key=args.dataset,
        batch_size=args.batch_size,
        augment=not args.no_augment,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")
    print(f"Num classes: {num_classes}")

    model = CharCNN(num_classes=num_classes, in_channels=config.IN_CHANNELS).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    ckpt_path = config.checkpoint_path(args.dataset)

    print(f"\n{'='*60}\nStarting training for {args.epochs} epochs\n{'='*60}")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "in_channels": config.IN_CHANNELS,
                "class_names": class_names,
                "dataset": args.dataset,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f}) to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no improvement for {args.patience} epochs).")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save training history
    hist_path = config.history_path(args.dataset)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")

    # Final test evaluation using best checkpoint
    print(f"\n{'='*60}\nEvaluating best checkpoint on TEST set\n{'='*60}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Append test results to history file
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
