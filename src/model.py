"""
src/model.py
=============
CNN architecture for handwritten character recognition.

CharCNN: A compact, well-regularized convolutional network suitable for
28x28 grayscale character images (MNIST / EMNIST). Uses Conv-BN-ReLU blocks,
max pooling, and dropout for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, optionally followed by MaxPool."""

    def __init__(self, in_ch, out_ch, pool=True, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class CharCNN(nn.Module):
    """
    CNN for 28x28x1 character images.

    Architecture:
        Block1: Conv(1->32)  + Conv(32->32)  -> pool -> 14x14
        Block2: Conv(32->64) + Conv(64->64)  -> pool -> 7x7
        Block3: Conv(64->128)                -> pool -> 3x3
        FC: 128*3*3 -> 256 -> num_classes

    Roughly ~1.2M parameters - trains fast on CPU and GPU.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.4):
        super().__init__()

        self.block1a = ConvBlock(in_channels, 32, pool=False)
        self.block1b = ConvBlock(32, 32, pool=True)         # 28 -> 14

        self.block2a = ConvBlock(32, 64, pool=False)
        self.block2b = ConvBlock(64, 64, pool=True)          # 14 -> 7

        self.block3 = ConvBlock(64, 128, pool=True)          # 7 -> 3 (floor)

        self.dropout = nn.Dropout(dropout)
        self.flatten_dim = 128 * 3 * 3

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1a(x)
        x = self.block1b(x)
        x = self.block2a(x)
        x = self.block2b(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Return softmax probabilities for input batch x."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = CharCNN(num_classes=47)
    dummy = torch.randn(8, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
