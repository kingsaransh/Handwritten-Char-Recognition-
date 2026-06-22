"""
src/crnn_train.py
===================
Trains the CRNN (src/crnn_model.py) for word-level handwriting recognition.

Since a dedicated word-level handwriting dataset (e.g. IAM Words) requires manual
registration and download, this script builds a SYNTHETIC word dataset by randomly
sampling and horizontally concatenating individual EMNIST character images. This
lets the CRNN + CTC pipeline be demonstrated and trained completely end-to-end with
zero extra downloads -- and the resulting model genuinely learns to read short
handwritten "words" formed from real handwritten character strokes.

To move to a real dataset such as IAM Words/Lines:
  1. Download IAM Words (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
  2. Replace `SyntheticWordDataset` below with a Dataset that returns (image, label_string)
     pairs from IAM, resized to the same (img_height, variable_width) convention.
  3. Everything else (model, CTC loss, training loop, decoding) stays the same.

Usage:
    python src/crnn_train.py --epochs 10 --words-per-epoch 5000
"""

import sys
import os
import argparse
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

import config
from src.crnn_model import CRNN, ctc_greedy_decode

# Character set for synthetic words: digits + uppercase letters (36 classes)
# (kept simple/legible; extend with lowercase if desired using EMNIST 'byclass')
CHAR_SET = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_SET)}  # 0-indexed; CTC blank added separately at idx 0 of model output

IMG_HEIGHT = 32
MIN_WORD_LEN = 3
MAX_WORD_LEN = 7


def _emnist_orientation_fix(img):
    img = TF.rotate(img, -90)
    img = TF.hflip(img)
    return img


class _CharacterBank:
    """Loads EMNIST 'byclass' once and indexes images by character for fast sampling."""

    def __init__(self):
        tf = transforms.Compose([
            transforms.Lambda(_emnist_orientation_fix),
            transforms.ToTensor(),
        ])
        print("Loading EMNIST 'byclass' split for synthetic word generation "
              "(this may download ~500MB on first run)...")
        self.dataset = datasets.EMNIST(root=config.DATA_DIR, split="byclass", train=True,
                                        download=True, transform=tf)

        # EMNIST byclass label mapping: 0-9 digits, 10-35 uppercase, 36-61 lowercase
        self.by_char = {c: [] for c in CHAR_SET}
        print("Indexing characters by class (one pass over labels)...")
        targets = self.dataset.targets.numpy()
        for idx in range(36):  # only digits + uppercase, matching CHAR_SET
            char = CHAR_SET[idx]
            matches = np.where(targets == idx)[0]
            self.by_char[char] = matches.tolist()

    def sample_image(self, char: str) -> torch.Tensor:
        indices = self.by_char[char]
        idx = random.choice(indices)
        img, _ = self.dataset[idx]   # (1, 28, 28) tensor in [0,1]
        return img.squeeze(0)        # (28, 28)


class SyntheticWordDataset(Dataset):
    """
    Generates synthetic "word" images on the fly by concatenating random EMNIST
    character images horizontally, with small random spacing/jitter.

    Each sample: (image_tensor (1, H, W), label_string)
    """

    def __init__(self, char_bank: _CharacterBank, n_samples: int,
                 min_len=MIN_WORD_LEN, max_len=MAX_WORD_LEN, img_height=IMG_HEIGHT):
        self.char_bank = char_bank
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self.img_height = img_height

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        word_len = random.randint(self.min_len, self.max_len)
        chars = [random.choice(CHAR_SET) for _ in range(word_len)]
        label = "".join(chars)

        char_imgs = []
        for c in chars:
            img28 = self.char_bank.sample_image(c)  # (28, 28) in [0,1]
            # Resize to img_height while keeping aspect ratio (square chars here, so same)
            img = img28.unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
            img = torch.nn.functional.interpolate(
                img, size=(self.img_height, self.img_height), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            char_imgs.append(img)
            # small random gap of background (black) between characters
            gap_w = random.randint(1, 4)
            char_imgs.append(torch.zeros(self.img_height, gap_w))

        word_img = torch.cat(char_imgs[:-1], dim=1)  # drop trailing gap, concat along width
        word_img = word_img.unsqueeze(0)             # (1, H, W)

        # normalize like training data (mean/std)
        word_img = (word_img - config.NORMALIZE_MEAN) / config.NORMALIZE_STD

        return word_img, label


def collate_fn(batch):
    """Pads variable-width images to the max width in the batch, builds CTC targets."""
    images, labels = zip(*batch)

    max_w = max(img.shape[2] for img in images)
    padded_imgs = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w), value=0)
        padded_imgs.append(img)
    images_batch = torch.stack(padded_imgs)  # (B, 1, H, max_W)

    targets = []
    target_lengths = []
    for label in labels:
        idxs = [CHAR_TO_IDX[c] + 1 for c in label]  # +1 because 0 is reserved for CTC blank
        targets.extend(idxs)
        target_lengths.append(len(idxs))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images_batch, targets, target_lengths, labels


def char_error_rate(pred: str, target: str) -> float:
    """Levenshtein-distance based character error rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(target) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(pred)][len(target)] / len(target)


def train_crnn(epochs=10, words_per_epoch=5000, batch_size=32, lr=1e-3, val_words=500):
    device = config.DEVICE
    print(f"Using device: {device}")

    char_bank = _CharacterBank()

    train_ds = SyntheticWordDataset(char_bank, n_samples=words_per_epoch)
    val_ds = SyntheticWordDataset(char_bank, n_samples=val_words)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = CRNN(num_classes=len(CHAR_SET), img_height=IMG_HEIGHT).to(device)
    print(f"CRNN parameters: {model.count_parameters():,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "crnn_word_best.pt")
    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, targets, target_lengths, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # (T, B, C+1)

            T = log_probs.size(0)
            B = log_probs.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = epoch_loss / len(train_loader)

        # Validation: average character error rate
        model.eval()
        total_cer = 0.0
        n_val = 0
        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:
                images = images.to(device)
                log_probs = model(images)
                decoded = ctc_greedy_decode(log_probs, CHAR_SET)
                for pred, true in zip(decoded, labels):
                    total_cer += char_error_rate(pred, true)
                    n_val += 1
        avg_cer = total_cer / max(n_val, 1)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | "
              f"Val CER: {avg_cer:.4f} | Time: {elapsed:.1f}s")

        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(CHAR_SET),
                "char_set": CHAR_SET,
                "img_height": IMG_HEIGHT,
                "val_cer": avg_cer,
            }, ckpt_path)
            print(f"  -> New best model saved (CER={avg_cer:.4f}) to {ckpt_path}")

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")

    # Show a few example predictions
    print("\nSample predictions on validation words:")
    model.eval()
    images, targets, target_lengths, labels = next(iter(val_loader))
    with torch.no_grad():
        log_probs = model(images.to(device))
        decoded = ctc_greedy_decode(log_probs, CHAR_SET)
    for pred, true in list(zip(decoded, labels))[:10]:
        marker = "OK" if pred == true else "X"
        print(f"  [{marker}] True: {true:10s} Pred: {pred}")


def main():
    parser = argparse.ArgumentParser(description="Train CRNN for word-level recognition")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--words-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_crnn(epochs=args.epochs, words_per_epoch=args.words_per_epoch,
               batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
