"""Inference-time preprocessing for the BUSI classifier.

This module is the single source of truth for the transforms applied to an
image before it reaches the model. It must stay identical to the `eval_tf`
used in the training notebook, otherwise predictions in production will
drift from the ones measured during evaluation (training/serving skew).
"""

from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms as T

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics (pretrained ResNet-18)
STD = [0.229, 0.224, 0.225]
CLASSES = ["benign", "malignant", "normal"]

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def preprocess(img: Image.Image | bytes) -> torch.Tensor:
    """Turn a PIL image (or raw image bytes) into a (1, 3, H, W) tensor."""
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))
    return eval_tf(img.convert("RGB")).unsqueeze(0)
