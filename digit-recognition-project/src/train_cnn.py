"""
train_cnn.py
------------
Trains a small Convolutional Neural Network (CNN) on MNIST using Keras.

CNNs generally outperform classical ML baselines on image tasks because
convolutional filters can learn spatial patterns (edges, curves, loops)
that make up handwritten digits, rather than treating every pixel as an
independent feature.

Usage:
    python src/train_cnn.py
"""

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

from data_loader import load_for_cnn


def build_cnn():
    """Builds a small but effective CNN architecture.

    Layers explained:
      - Conv2D: learns local visual features (edges, strokes)
      - MaxPooling2D: downsamples, keeping the strongest signal
      - Dropout: randomly disables neurons during training to reduce
        overfitting
      - Dense: fully-connected layers for final classification
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),  # 10 digit classes
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(epochs=15, batch_size=128):
    print("Loading data for CNN...")
    X_train, y_train, X_test, y_test = load_for_cnn()

    model = build_cnn()
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print("\nTraining CNN...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    # Evaluate on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n=== CNN results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=3))

    model.save("models/cnn_model.h5")
    print("Saved CNN model to models/cnn_model.h5")

    return model, history, {"accuracy": acc, "f1": f1, "y_pred": y_pred, "y_true": y_true}


if __name__ == "__main__":
    train_and_evaluate()
