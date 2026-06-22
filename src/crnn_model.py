"""
src/crnn_model.py
====================
CRNN (Convolutional Recurrent Neural Network) for full WORD / LINE recognition.

This is the extendable sequence-modeling path mentioned in the project brief:
instead of segmenting a word into individual characters (see src/predict.py's
`predict_word`), a CRNN reads a variable-width image of an entire word/line in
one pass and outputs a character sequence directly, trained end-to-end with
CTC (Connectionist Temporal Classification) loss. This avoids segmentation
errors common with touching/cursive handwriting.

Architecture:
    CNN backbone (feature extraction over the image width)
      -> reshape feature map into a sequence of column-vectors
      -> Bidirectional LSTM (sequence modeling, context from both directions)
      -> Linear classifier per time-step (character probabilities + CTC blank)
      -> CTC decoding (greedy or beam search) to get the final string

This module is self-contained and works on its own synthetic data (concatenated
EMNIST characters) so you can try it immediately without a separate word-level
dataset such as IAM. To train on a real handwriting dataset (e.g. IAM Words/Lines),
swap out `SyntheticWordDataset` (in crnn_train.py) for a loader over that dataset,
keeping the same (image, label_string) interface.

Usage:
    python src/crnn_model.py --selftest         # quick architecture smoke-test
    python src/crnn_train.py                    # train on synthetic concatenated words
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CNN + BiLSTM + CTC head for variable-length character sequence recognition.

    Input:  (B, 1, H, W) grayscale word/line image, H fixed (e.g. 32), W variable.
    Output: (T, B, num_classes+1) log-probabilities over time steps T (CTC format),
             where class index 0 is reserved for the CTC "blank" token.
    """

    def __init__(self, num_classes: int, img_height: int = 32, hidden_size: int = 256):
        super().__init__()
        self.num_classes = num_classes  # excludes blank
        self.img_height = img_height

        # CNN feature extractor: downsamples height to 1, keeps width informative
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                              # H/2 x W/2

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                              # H/4 x W/4

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                                            # H/8 x W/4

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                                            # H/16 x W/4

            nn.Conv2d(512, 512, 2, padding=(0, 1)), nn.BatchNorm2d(512), nn.ReLU(),
        )

        # After the CNN, height should be reduced to 1 for img_height=32
        cnn_out_height = img_height // 16 - 1
        assert cnn_out_height >= 1, "img_height too small for this CNN architecture (use >= 32)"

        self.map_to_seq = nn.Linear(512 * cnn_out_height, hidden_size)

        self.rnn1 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=False)

        # +1 for CTC blank token at index 0
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x)                       # (B, C, H', W')
        b, c, h, w = conv.size()
        conv = conv.permute(3, 0, 1, 2)           # (W', B, C, H') -- width becomes time dimension
        conv = conv.reshape(w, b, c * h)          # (T=W', B, C*H')

        seq = self.map_to_seq(conv)               # (T, B, hidden_size)

        seq, _ = self.rnn1(seq)
        seq, _ = self.rnn2(seq)

        out = self.fc(seq)                        # (T, B, num_classes+1)
        log_probs = out.log_softmax(dim=2)
        return log_probs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def ctc_greedy_decode(log_probs: torch.Tensor, class_names):
    """
    Greedy CTC decode: takes argmax per timestep, collapses repeats, removes blanks.

    Args:
        log_probs: (T, B, num_classes+1) tensor (output of CRNN.forward)
        class_names: list of length num_classes (index 0 in log_probs is blank,
                      so class_names[i] corresponds to log_probs index i+1)

    Returns:
        List of decoded strings, one per batch element.
    """
    # (T, B)
    preds = log_probs.argmax(dim=2)
    preds = preds.permute(1, 0)  # (B, T)

    results = []
    for seq in preds:
        seq = seq.tolist()
        decoded = []
        prev = -1
        for token in seq:
            if token != 0 and token != prev:   # skip blank (0) and repeated tokens
                decoded.append(class_names[token - 1])
            prev = token
        results.append("".join(decoded))
    return results


def _selftest():
    """Quick smoke test verifying shapes and a forward pass."""
    num_classes = 36  # e.g. 26 letters + 10 digits
    model = CRNN(num_classes=num_classes, img_height=32)
    print(f"CRNN parameters: {model.count_parameters():,}")

    dummy = torch.randn(4, 1, 32, 160)  # batch of 4, height 32, width 160 (variable in practice)
    log_probs = model(dummy)
    print(f"Output shape (T, B, C+1): {log_probs.shape}")

    class_names = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
    decoded = ctc_greedy_decode(log_probs, class_names)
    print(f"Decoded (random weights, gibberish expected): {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        _selftest()
