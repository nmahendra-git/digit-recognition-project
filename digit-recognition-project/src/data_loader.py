"""
data_loader.py
--------------
Loads the MNIST handwritten digit dataset and prepares it for two kinds
of models:
  1. Classical ML models (SVM, MLP from scikit-learn) -> need flattened,
     scaled vectors of shape (n_samples, 784).
  2. CNNs (Keras) -> need images reshaped to (n_samples, 28, 28, 1),
     scaled to [0, 1], with one-hot encoded labels.

Run this file directly to sanity-check that the dataset loads correctly:
    python src/data_loader.py
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_raw_mnist():
    """Downloads (once, then cached) and returns the raw MNIST arrays.

    Returns:
        (x_train, y_train), (x_test, y_test)
        x_*: uint8 arrays of shape (N, 28, 28), pixel values 0-255
        y_*: uint8 arrays of shape (N,), digit labels 0-9
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def load_for_classical_ml(sample_size=None, random_state=42):
    """Prepares MNIST for scikit-learn models (SVM / MLP).

    Flattens each 28x28 image into a 784-length vector and scales pixel
    values to the [0, 1] range.

    Args:
        sample_size: if set, randomly subsamples this many training
            examples. Useful because SVMs are slow on the full 60k
            training set. Recommended: 5000-10000 for a laptop.
        random_state: seed for reproducible subsampling.

    Returns:
        X_train, y_train, X_test, y_test (all NumPy arrays)
    """
    (x_train, y_train), (x_test, y_test) = load_raw_mnist()

    X_train = x_train.reshape(len(x_train), -1).astype("float32") / 255.0
    X_test = x_test.reshape(len(x_test), -1).astype("float32") / 255.0

    if sample_size is not None and sample_size < len(X_train):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_train), size=sample_size, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test


def load_for_cnn():
    """Prepares MNIST for a Keras CNN.

    Returns:
        X_train, y_train, X_test, y_test
        X_*: float32 arrays of shape (N, 28, 28, 1), scaled to [0, 1]
        y_*: one-hot encoded arrays of shape (N, 10)
    """
    (x_train, y_train), (x_test, y_test) = load_raw_mnist()

    X_train = x_train.astype("float32") / 255.0
    X_test = x_test.astype("float32") / 255.0

    X_train = np.expand_dims(X_train, -1)  # (N, 28, 28, 1)
    X_test = np.expand_dims(X_test, -1)

    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    return X_train, y_train_cat, X_test, y_test_cat


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_raw_mnist()
    print(f"Train images: {x_train.shape}, Train labels: {y_train.shape}")
    print(f"Test images:  {x_test.shape}, Test labels:  {y_test.shape}")
    print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
    print(f"Classes: {np.unique(y_train)}")
