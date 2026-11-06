"""Classify your own clothing photo with the trained Fashion-MNIST CNN.

Usage:
    python predict.py path/to/photo.jpg

Fashion-MNIST images are light items on a black background, 28x28, grayscale.
A normal photo is the opposite (dark item on a light background), so the image
is inverted when its background is light. Without this step most real photos
get the wrong class.
"""
import sys

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

IMG_SIZE = 28
MODEL_PATH = "model/fashion_cnn.keras"

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",
]


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")

    # Pad to a square first so the item is not stretched
    img = ImageOps.pad(img, (max(img.size),) * 2, color=img.getpixel((0, 0)))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Light background -> invert to match the dataset (item light, background dark)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:
        arr = 1.0 - arr

    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict(image_path, model):
    probs = model.predict(preprocess_image(image_path), verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(LABELS[i], float(probs[i])) for i in top]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py path/to/photo.jpg")

    model = keras.models.load_model(MODEL_PATH)
    for label, p in predict(sys.argv[1], model):
        print(f"{label:<12} {p:6.1%}")
