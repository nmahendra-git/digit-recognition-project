"""
train_baseline.py
------------------
Trains two classical machine learning baselines on MNIST:
  1. Support Vector Machine (SVM)
  2. Multi-Layer Perceptron (MLP), i.e. a plain feed-forward neural net

These baselines are useful because they show you how much a CNN's
convolutional structure actually helps versus a "generic" model.

Usage:
    python src/train_baseline.py
"""

import time
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_loader import load_for_classical_ml


def train_svm(X_train, y_train):
    print("\nTraining SVM (this can take a few minutes)...")
    start = time.time()
    # RBF kernel SVM. C and gamma are reasonable defaults for MNIST.
    model = SVC(kernel="rbf", C=5, gamma="scale")
    model.fit(X_train, y_train)
    print(f"SVM trained in {time.time() - start:.1f}s")
    return model


def train_mlp(X_train, y_train):
    print("\nTraining MLP (feed-forward neural net)...")
    start = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=30,
        random_state=42,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    print(f"MLP trained in {time.time() - start:.1f}s")
    return model


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== {name} results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    return {"accuracy": acc, "f1": f1, "y_pred": y_pred}


if __name__ == "__main__":
    # SVMs get slow on the full 60k training set on a laptop,
    # so we subsample for the SVM. Feel free to increase this
    # if your machine can handle it, or set to None for the full set.
    SVM_SAMPLE_SIZE = 8000

    print("Loading data for SVM (subsampled)...")
    X_train_s, y_train_s, X_test, y_test = load_for_classical_ml(
        sample_size=SVM_SAMPLE_SIZE
    )
    svm_model = train_svm(X_train_s, y_train_s)
    svm_results = evaluate_model(svm_model, X_test, y_test, "SVM")
    joblib.dump(svm_model, "models/svm_model.pkl")
    print("Saved SVM model to models/svm_model.pkl")

    print("\nLoading full data for MLP...")
    X_train, y_train, X_test, y_test = load_for_classical_ml()
    mlp_model = train_mlp(X_train, y_train)
    mlp_results = evaluate_model(mlp_model, X_test, y_test, "MLP")
    joblib.dump(mlp_model, "models/mlp_model.pkl")
    print("Saved MLP model to models/mlp_model.pkl")
