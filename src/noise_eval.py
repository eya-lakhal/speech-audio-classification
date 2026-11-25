"""Noise robustness: evaluate every trained model on the test set corrupted
with background noise at several SNRs.

Usage:  python -m src.noise_eval

The same noisy test sets (same noise excerpts, same seed) are used for all
models, so the curves are directly comparable. Noise comes from the *test*
part (last 30 %) of each `_background_noise_` recording; the CNN augmentation
only ever used the first 70 %.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import WORDS
from src.evaluate import RESULTS_DIR, metrics, plot_confusion, top_confusions
from src.features import CACHE_DIR, classic_features_batch, load_split_waveforms, make_noisy_set

SNRS = [None, 20, 10, 5, 0]  # None = clean
MODELS_DIR = Path("models")
LABELS = {
    "random_forest": "Random Forest (MFCC)",
    "svm": "SVM (MFCC)",
    "cnn_noaug": "CNN log-mel, no augmentation",
    "cnn": "CNN log-mel, augmented",
}


def snr_name(snr) -> str:
    return "clean" if snr is None else f"{snr} dB"


def noisy_test(x_test: np.ndarray, snr) -> np.ndarray:
    return x_test if snr is None else make_noisy_set(x_test, snr, seed=1234)


def load_models() -> dict:
    models = {}
    for name in ("random_forest", "svm"):
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            models[name] = ("classic", joblib.load(p))
    for name in ("cnn_noaug", "cnn"):
        p = MODELS_DIR / f"{name}.keras"
        if p.exists():
            import keras
            import src.train_cnn  # noqa: F401  registers the LogMel layer
            models[name] = ("cnn", keras.models.load_model(p))
    return models


def main() -> None:
    x_test, y_test = load_split_waveforms("test")
    models = load_models()
    print("Models:", ", ".join(models))

    rows = []
    for snr in SNRS:
        x = noisy_test(x_test, snr)
        feats = None
        for name, (kind, model) in models.items():
            if kind == "classic":
                if feats is None:
                    cache = CACHE_DIR / f"classic_test_{snr_name(snr).replace(' ', '')}.npy"
                    if cache.exists():
                        feats = np.load(cache)
                    else:
                        feats = classic_features_batch(x, desc=f"Features {snr_name(snr)}")
                        np.save(cache, feats)
                y_pred = model.predict(feats)
            else:
                probs = model.predict(x, batch_size=256, verbose=0)
                y_pred = np.array(WORDS)[probs.argmax(axis=1)]
            m = metrics(y_test, y_pred)
            rows.append({"model": name, "snr": snr_name(snr), **m})
            print(f"{snr_name(snr):>6}  {name:<14} acc {m['accuracy']:.4f}  F1 {m['f1_macro']:.4f}")

            if snr == 0:  # hardest condition: look at what goes wrong
                plot_confusion(y_test, y_pred, f"{LABELS[name]} - 0 dB SNR",
                               RESULTS_DIR / f"confusion_{name}_0dB.png")
                rows[-1]["top_confusions_0dB"] = top_confusions(y_test, y_pred)

    df = pd.DataFrame(rows)
    df.drop(columns="top_confusions_0dB", errors="ignore").to_csv(
        RESULTS_DIR / "noise_results.csv", index=False)
    with open(RESULTS_DIR / "noise_confusions_0dB.json", "w") as f:
        json.dump({r["model"]: r["top_confusions_0dB"] for r in rows
                   if "top_confusions_0dB" in r}, f, indent=2, default=str)
    plot_curves(df)


def plot_curves(df: pd.DataFrame) -> None:
    order = [snr_name(s) for s in SNRS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    markers = {"random_forest": "s", "svm": "^", "cnn_noaug": "o", "cnn": "D"}
    for ax, metric, title in zip(axes, ("accuracy", "f1_macro"), ("Accuracy", "Macro F1-score")):
        for name, g in df.groupby("model", sort=False):
            g = g.set_index("snr").loc[order]
            ax.plot(order, g[metric] * 100, marker=markers.get(name, "o"), label=LABELS[name])
        ax.set_title(f"{title} vs SNR")
        ax.set_xlabel("Test condition (signal-to-noise ratio)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_snr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
