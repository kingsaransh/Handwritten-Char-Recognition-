"""
config.py
=========
Central configuration for the Handwritten Character Recognition project.
Edit values here to change dataset, model, or training behaviour everywhere at once.
"""

import os
import torch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------------------
# Options:
#   "mnist"            -> 10 classes, digits 0-9
#   "emnist_letters"    -> 26 classes, letters A-Z (case-insensitive merged)
#   "emnist_balanced"    -> 47 classes, digits + upper + lower (balanced split)
DATASET = os.environ.get("HCR_DATASET", "emnist_balanced")

DATASET_CHOICES = {
    "mnist": {
        "name": "MNIST",
        "num_classes": 10,
        "torchvision_name": "MNIST",
        "split_arg": None,
    },
    "emnist_letters": {
        "name": "EMNIST-Letters",
        "num_classes": 26,
        "torchvision_name": "EMNIST",
        "split_arg": "letters",
    },
    "emnist_balanced": {
        "name": "EMNIST-Balanced",
        "num_classes": 47,
        "torchvision_name": "EMNIST",
        "split_arg": "balanced",
    },
}

# EMNIST "balanced" split label mapping (47 classes) -> characters
# Order follows the official EMNIST mapping file (ASCII codes for balanced split)
EMNIST_BALANCED_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

EMNIST_LETTERS_CLASSES = [chr(ord('A') + i) for i in range(26)]
MNIST_CLASSES = [str(i) for i in range(10)]


def get_class_names(dataset_key: str):
    if dataset_key == "mnist":
        return MNIST_CLASSES
    elif dataset_key == "emnist_letters":
        return EMNIST_LETTERS_CLASSES
    elif dataset_key == "emnist_balanced":
        return EMNIST_BALANCED_CLASSES
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


# ----------------------------------------------------------------------------
# Image / preprocessing
# ----------------------------------------------------------------------------
IMG_SIZE = 28            # native size for MNIST/EMNIST
IN_CHANNELS = 1          # grayscale
NORMALIZE_MEAN = 0.1307
NORMALIZE_STD = 0.3081

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
EARLY_STOP_PATIENCE = 5
SEED = 42

# ----------------------------------------------------------------------------
# Model checkpoint filenames
# ----------------------------------------------------------------------------
def checkpoint_path(dataset_key: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"cnn_{dataset_key}_best.pt")


def history_path(dataset_key: str) -> str:
    return os.path.join(OUTPUT_DIR, f"history_{dataset_key}.json")
