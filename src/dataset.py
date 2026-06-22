"""
src/dataset.py
===============
Dataset loading & preprocessing for MNIST / EMNIST.

EMNIST images, as stored by torchvision, are rotated 90 degrees and flipped
relative to MNIST's orientation. We correct for this with a transform so that
all characters appear "right way up" regardless of dataset chosen.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def _patch_mnist_mirror():
    """
    torchvision's default MNIST mirrors (yann.lecun.com / ossci-datasets S3) are
    occasionally unreachable behind restrictive firewalls/proxies. This swaps in
    a reliable GitHub-hosted mirror (https://github.com/fgnt/mnist) as a fallback
    so `download=True` keeps working in locked-down network environments.
    Safe to call multiple times; has no effect once already patched.
    """
    mirror = "https://raw.githubusercontent.com/fgnt/mnist/master"
    try:
        current_urls = [r[0] for r in datasets.MNIST.resources]
        if not any(mirror in u for u in current_urls):
            datasets.MNIST.mirrors = [mirror + "/"]
    except Exception:
        pass  # If torchvision's internals change, just fall back to default behaviour.


_patch_mnist_mirror()


def _emnist_orientation_fix(img):
    """EMNIST samples are transposed + vertically flipped compared to MNIST.
    This rotates -90 and flips horizontally to match natural reading orientation."""
    import torchvision.transforms.functional as F
    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


def get_transforms(dataset_key: str, augment: bool = False):
    """Return train/test torchvision transforms for the chosen dataset."""
    base_ops = []

    if dataset_key.startswith("emnist"):
        base_ops.append(transforms.Lambda(_emnist_orientation_fix))

    if augment:
        train_ops = base_ops + [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,)),
        ]
    else:
        train_ops = base_ops + [
            transforms.ToTensor(),
            transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,)),
        ]

    test_ops = base_ops + [
        transforms.ToTensor(),
        transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,)),
    ]

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def load_datasets(dataset_key: str = None, augment: bool = True, val_split: float = 0.1):
    """
    Downloads (if needed) and returns train/val/test datasets for the given dataset key.

    Returns:
        train_ds, val_ds, test_ds, num_classes, class_names
    """
    dataset_key = dataset_key or config.DATASET
    if dataset_key not in config.DATASET_CHOICES:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choices: {list(config.DATASET_CHOICES)}")

    info = config.DATASET_CHOICES[dataset_key]
    train_tf, test_tf = get_transforms(dataset_key, augment=augment)
    class_names = config.get_class_names(dataset_key)

    if dataset_key == "mnist":
        train_full = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=test_tf)
    else:
        split = info["split_arg"]
        train_full = datasets.EMNIST(root=config.DATA_DIR, split=split, train=True,
                                      download=True, transform=train_tf)
        test_ds = datasets.EMNIST(root=config.DATA_DIR, split=split, train=False,
                                   download=True, transform=test_tf)
        # EMNIST 'letters' split labels are 1-indexed (1-26); shift to 0-indexed.
        if split == "letters":
            train_full = _ShiftedLabelDataset(train_full, shift=-1)
            test_ds = _ShiftedLabelDataset(test_ds, shift=-1)

    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    return train_ds, val_ds, test_ds, info["num_classes"], class_names


class _ShiftedLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset and shifts integer labels by a constant (e.g. 1-indexed -> 0-indexed)."""

    def __init__(self, base_dataset, shift: int):
        self.base = base_dataset
        self.shift = shift

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y + self.shift


def get_dataloaders(dataset_key: str = None, batch_size: int = None, augment: bool = True,
                     val_split: float = 0.1, num_workers: int = None):
    """Convenience wrapper returning ready-to-use DataLoaders."""
    dataset_key = dataset_key or config.DATASET
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    train_ds, val_ds, test_ds, num_classes, class_names = load_datasets(
        dataset_key, augment=augment, val_split=val_split
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, num_classes, class_names


if __name__ == "__main__":
    # Quick smoke test: load a small batch and print shapes.
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"Dataset: {config.DATASET} | classes: {num_classes}")
    print(f"Batch shape: {x.shape}, labels shape: {y.shape}")
    print(f"Label range: {y.min().item()} - {y.max().item()}")
    print(f"Class names ({len(class_names)}): {class_names}")
