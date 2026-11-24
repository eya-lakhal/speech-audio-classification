"""Shared evaluation helpers: metrics, confusion matrices, error analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import WORDS

RESULTS_DIR = Path("results")


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def plot_confusion(y_true, y_pred, title: str, path: Path, labels=WORDS) -> None:
    """Row-normalised confusion matrix (each row sums to 100 %)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 50 else "black")
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(y_true, y_pred, k: int = 5, labels=WORDS) -> list[tuple[str, str, int, float]]:
    """Most frequent (true -> predicted) mistakes with their count and % of the true class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    support = np.bincount([labels.index(t) for t in y_true], minlength=len(labels))
    pairs = [(labels[i], labels[j], int(cm[i, j]), 100 * cm[i, j] / support[i])
             for i in range(len(labels)) for j in range(len(labels)) if cm[i, j] > 0]
    return sorted(pairs, key=lambda p: -p[2])[:k]


def format_confusions(pairs) -> str:
    return "\n".join(f"  {t:>5} -> {p:<5} {n:4d} ({pct:.1f} % of '{t}')" for t, p, n, pct in pairs)
