"""
src/predict.py
================
Inference utilities for predicting handwritten characters from:
  - a single image file (PNG/JPG of one character)
  - a numpy array / PIL image (e.g. from a web canvas)
  - a full image containing a WORD or SENTENCE (multiple characters),
    using contour-based segmentation to isolate and classify each character.

Usage:
    python src/predict.py --image path/to/char.png --dataset emnist_balanced
    python src/predict.py --image path/to/word.png --dataset emnist_balanced --segment
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import config
from src.model import CharCNN


def load_inference_model(dataset_key: str, device=None):
    device = device or config.DEVICE
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
    return model, checkpoint["class_names"], device


def preprocess_char_image(img: np.ndarray, target_size: int = 28, pad_ratio: float = 0.2) -> np.ndarray:
    """
    Convert an arbitrary single-character image (grayscale, any size, any background)
    into a normalized 28x28 array matching MNIST/EMNIST style:
      - white digit/character on black background
      - centered, with padding
      - normalized to [0, 1] then standardized

    Args:
        img: 2D numpy array, grayscale, values 0-255.
        target_size: output square size (28 for MNIST/EMNIST-trained models).
        pad_ratio: fraction of target_size to use as padding around the bounding box.

    Returns:
        normalized 28x28 float32 numpy array ready for tensor conversion.
    """
    img = img.astype(np.uint8)

    # Ensure white-on-black like MNIST/EMNIST. If background is bright (mean high), invert.
    if img.mean() > 127:
        img = 255 - img

    # Threshold to clean up noise / anti-aliasing artifacts
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Find bounding box of the character (non-zero pixels)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Blank image fallback
        return np.zeros((target_size, target_size), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Resize, preserving aspect ratio, to fit inside (target_size - padding)
    pad = int(target_size * pad_ratio)
    inner_size = target_size - 2 * pad
    scale = inner_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste into a target_size x target_size black canvas, centered
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas.astype(np.float32)


def array_to_tensor(arr_28x28: np.ndarray) -> torch.Tensor:
    """Convert a 28x28 [0,255] array into a normalized (1,1,28,28) tensor."""
    arr = arr_28x28 / 255.0
    arr = (arr - config.NORMALIZE_MEAN) / config.NORMALIZE_STD
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_single_character(model, class_names, img: np.ndarray, device, top_k: int = 3):
    """
    Predict the character class for a single pre-cropped character image.

    Returns:
        dict with 'prediction', 'confidence', and 'top_k' list of (char, prob).
    """
    processed = preprocess_char_image(img)
    tensor = array_to_tensor(processed).to(device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1][:top_k]
    top_k_results = [(class_names[i], float(probs[i])) for i in top_indices]

    return {
        "prediction": class_names[top_indices[0]],
        "confidence": float(probs[top_indices[0]]),
        "top_k": top_k_results,
        "processed_image": processed,
    }


def segment_characters(img: np.ndarray, min_area: int = 30):
    """
    Segment a word/sentence image into individual character bounding boxes
    using contour detection. Returns boxes sorted left-to-right.

    Args:
        img: grayscale image, white text on dark OR dark text on light background.
        min_area: minimum contour area to keep (filters noise specks).

    Returns:
        List of (x, y, w, h) bounding boxes, sorted by x-coordinate (reading order).
    """
    img = img.astype(np.uint8)
    if img.mean() > 127:
        inv = 255 - img
    else:
        inv = img.copy()

    _, thresh = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)

    # Dilate slightly to merge broken strokes within a single character (e.g. 'i' dot)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes


@torch.no_grad()
def predict_word(model, class_names, img: np.ndarray, device, min_area: int = 30,
                  merge_gap_ratio: float = 0.0):
    """
    Predict a sequence of characters from a word/sentence image via segmentation
    + per-character classification. This is the simple "segment-then-classify"
    approach -- see README for the CRNN sequence-modeling extension that avoids
    explicit segmentation.

    Returns:
        dict with 'text' (concatenated prediction) and 'characters' (per-char detail).
    """
    boxes = segment_characters(img, min_area=min_area)

    img_u8 = img.astype(np.uint8)
    results = []
    text = ""

    for (x, y, w, h) in boxes:
        # Add small margin around each character crop
        margin = max(2, int(0.1 * max(w, h)))
        y0 = max(0, y - margin)
        y1 = min(img_u8.shape[0], y + h + margin)
        x0 = max(0, x - margin)
        x1 = min(img_u8.shape[1], x + w + margin)
        char_crop = img_u8[y0:y1, x0:x1]

        result = predict_single_character(model, class_names, char_crop, device, top_k=1)
        results.append({
            "char": result["prediction"],
            "confidence": result["confidence"],
            "box": (int(x), int(y), int(w), int(h)),
        })
        text += result["prediction"]

    return {"text": text, "characters": results}


def predict_from_path(image_path: str, dataset_key: str, segment: bool = False, top_k: int = 3):
    """High-level helper: load an image from disk and run prediction."""
    model, class_names, device = load_inference_model(dataset_key)

    pil_img = Image.open(image_path).convert("L")
    img = np.array(pil_img)

    if segment:
        return predict_word(model, class_names, img, device)
    else:
        return predict_single_character(model, class_names, img, device, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a handwritten character/word image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--dataset", type=str, default=config.DATASET,
                         choices=list(config.DATASET_CHOICES.keys()))
    parser.add_argument("--segment", action="store_true",
                         help="Treat image as a word/sentence and segment into characters")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found at {args.image}")
        sys.exit(1)

    result = predict_from_path(args.image, args.dataset, segment=args.segment, top_k=args.top_k)

    if args.segment:
        print(f"\nPredicted text: '{result['text']}'")
        print("\nPer-character detail:")
        for c in result["characters"]:
            print(f"  '{c['char']}'  confidence={c['confidence']:.2%}  box={c['box']}")
    else:
        print(f"\nPrediction: '{result['prediction']}'  (confidence={result['confidence']:.2%})")
        print("\nTop-k predictions:")
        for char, prob in result["top_k"]:
            print(f"  '{char}': {prob:.2%}")


if __name__ == "__main__":
    main()
