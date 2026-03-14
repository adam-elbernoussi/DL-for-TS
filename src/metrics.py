"""
Unified evaluation utilities for the LSST classification project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


def evaluate(y_true, y_pred, class_names=None, model_name="Model"):
    """Print classification report and return a dict of key metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")
    print(f"=== {model_name} ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Weighted F1: {f1_w:.4f}")
    print(f"Macro F1:    {f1_m:.4f}")
    target_names = list(class_names) if class_names is not None else None
    print(classification_report(y_true, y_pred, target_names=target_names))
    return dict(accuracy=acc, weighted_f1=f1_w, macro_f1=f1_m)


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix",
                          figsize=(8, 7), normalize=True):
    """Plot a confusion matrix heatmap. Returns fig, ax."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = list(class_names) if class_names is not None else None
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def results_table(results_dict):
    """Pretty-print a comparison table from {model_name: metrics_dict}."""
    header = f"{'Model':<30} {'Acc':>8} {'W-F1':>8} {'M-F1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results_dict.items():
        print(f"{name:<30} {m['accuracy']:>8.4f} {m['weighted_f1']:>8.4f} {m['macro_f1']:>8.4f}")
