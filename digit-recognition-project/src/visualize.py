"""
visualize.py
------------
Helper functions to visualize MNIST digits, predictions, and confusion
matrices. Meant to be imported inside the Jupyter notebook, but can also
be run standalone.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_sample_digits(images, labels, n=10, title="Sample digits"):
    """Plots a row of sample digit images with their labels.

    Args:
        images: array of shape (N, 28, 28) or (N, 784)
        labels: array of shape (N,)
        n: how many samples to show
    """
    images = np.array(images)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    fig.suptitle(title)
    for i in range(n):
        axes[i].imshow(images[i], cmap="gray")
        axes[i].set_title(str(labels[i]))
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_predictions(images, y_true, y_pred, n=10, title="Predictions (green=correct, red=wrong)"):
    """Plots sample predictions, highlighting correct vs incorrect ones."""
    images = np.array(images)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    elif images.ndim == 4:  # (N, 28, 28, 1)
        images = images.reshape(images.shape[0], 28, 28)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    fig.suptitle(title)
    for i in range(n):
        axes[i].imshow(images[i], cmap="gray")
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"
        axes[i].set_title(f"T:{y_true[i]} P:{y_pred[i]}", color=color)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots a heatmap confusion matrix for the 10 digit classes."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.show()


def plot_training_history(history):
    """Plots training/validation accuracy and loss curves (Keras History object)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
