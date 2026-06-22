import sys
import os
import io
import base64
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import config
from src.predict import (
    load_inference_model,
    predict_single_character,
    predict_word,
)

app = Flask(__name__)

# Global state populated at startup (see main())
MODEL = None
CLASS_NAMES = None
DEVICE = None
DATASET_KEY = None


def decode_canvas_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL (from <canvas>.toDataURL()) into a grayscale numpy array."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(binary)).convert("L")
    return np.array(img)


@app.route("/")
def index():
    return render_template(
        "index.html",
        dataset_name=config.DATASET_CHOICES[DATASET_KEY]["name"],
        num_classes=len(CLASS_NAMES),
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    mode = payload.get("mode", "character")
    image_data = payload.get("image")

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    img = decode_canvas_image(image_data)

    if mode == "word":
        result = predict_word(MODEL, CLASS_NAMES, img, DEVICE)
        return jsonify({
            "mode": "word",
            "text": result["text"],
            "characters": [
                {"char": c["char"], "confidence": c["confidence"], "box": c["box"]}
                for c in result["characters"]
            ],
        })
    else:
        result = predict_single_character(MODEL, CLASS_NAMES, img, DEVICE, top_k=5)
        return jsonify({
            "mode": "character",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top_k": [{"char": c, "prob": p} for c, p in result["top_k"]],
        })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset": DATASET_KEY,
        "num_classes": len(CLASS_NAMES),
        "device": str(DEVICE),
    })


def main():
    global MODEL, CLASS_NAMES, DEVICE, DATASET_KEY

    parser = argparse.ArgumentParser(description="Run the handwritten character recognition web demo")
    parser.add_argument("--dataset", type=str, default=config.DATASET,
                         choices=list(config.DATASET_CHOICES.keys()))
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DATASET_KEY = args.dataset
    print(f"Loading model for dataset: {DATASET_KEY} ...")
    MODEL, CLASS_NAMES, DEVICE = load_inference_model(DATASET_KEY)
    print(f"Model loaded. {len(CLASS_NAMES)} classes. Device: {DEVICE}")
    print(f"Starting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
