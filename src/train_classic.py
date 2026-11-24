"""Classical baseline: hand-crafted features -> Random Forest and SVM.

Usage:  python -m src.train_classic

Hyper-parameters are chosen with GridSearchCV on the *official* validation
split (PredefinedSplit) rather than random K-fold: random folds would put the
same speaker in both the fitting and the scoring part, which inflates scores.
After the search, the best model is refit on train + val and evaluated once
on the test split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluate import RESULTS_DIR, format_confusions, metrics, plot_confusion, top_confusions
from src.features import CACHE_DIR, classic_features_batch, load_split_waveforms

MODELS_DIR = Path("models")


def load_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Classical features for a split, cached in data/cache/classic_<split>.npz."""
    cache = CACHE_DIR / f"classic_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    x_wave, y = load_split_waveforms(split)
    x = classic_features_batch(x_wave, desc=f"Features {split}")
    np.savez(cache, x=x, y=y)
    return x, y


def search_spaces() -> dict[str, tuple]:
    rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    rf_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 25],
        "min_samples_leaf": [1, 2],
    }
    # SVMs are sensitive to feature scales -> standardise first.
    svm = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    svm_grid = {
        "svc__C": [1, 10, 30],
        "svc__gamma": ["scale", 0.005],
    }
    return {"random_forest": (rf, rf_grid), "svm": (svm, svm_grid)}


def main() -> None:
    x_tr, y_tr = load_features("train")
    x_va, y_va = load_features("val")
    x_te, y_te = load_features("test")
    print(f"Features: train {x_tr.shape}, val {x_va.shape}, test {x_te.shape}")

    # -1 = always in the fitting part, 0 = the (single) validation fold
    x_trva = np.concatenate([x_tr, x_va])
    y_trva = np.concatenate([y_tr, y_va])
    fold = PredefinedSplit(np.r_[np.full(len(x_tr), -1), np.zeros(len(x_va))])

    MODELS_DIR.mkdir(exist_ok=True)
    summary = {}
    for name, (estimator, grid) in search_spaces().items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        search = GridSearchCV(estimator, grid, cv=fold, scoring="f1_macro",
                              n_jobs=1 if name == "random_forest" else -1,
                              refit=True, verbose=1)
        search.fit(x_trva, y_trva)
        elapsed = time.time() - t0

        y_pred = search.predict(x_te)
        test = metrics(y_te, y_pred)
        print(f"best params {search.best_params_}  val F1 {search.best_score_:.4f}")
        print(f"test accuracy {test['accuracy']:.4f}  test F1 {test['f1_macro']:.4f}  ({elapsed:.0f}s)")
        conf = top_confusions(y_te, y_pred)
        print("Most frequent confusions:\n" + format_confusions(conf))

        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        plot_confusion(y_te, y_pred, f"{name.replace('_', ' ').title()} - clean test",
                       RESULTS_DIR / f"confusion_{name}.png")
        summary[name] = {
            "best_params": search.best_params_,
            "val_f1_macro": float(search.best_score_),
            "test": test,
            "top_confusions": conf,
            "search_seconds": round(elapsed),
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classic_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
