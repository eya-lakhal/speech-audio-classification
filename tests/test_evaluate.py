"""Tests for evaluation helpers and Whisper text normalisation."""

import numpy as np

from src.data import WORDS
from src.evaluate import metrics, top_confusions
from src.whisper_eval import normalize_text, stratified_sample


def test_metrics_perfect_and_wrong():
    y = ["yes", "no", "up"]
    assert metrics(y, y) == {"accuracy": 1.0, "f1_macro": 1.0}
    assert metrics(y, ["no", "yes", "up"])["accuracy"] == 1 / 3


def test_top_confusions_counts_and_percent():
    y_true = ["go"] * 4 + ["no"] * 2
    y_pred = ["no", "no", "go", "go", "no", "go"]
    top = top_confusions(y_true, y_pred, k=2)
    assert top[0] == ("go", "no", 2, 50.0)   # 2 of the 4 'go' predicted 'no'
    assert top[1] == ("no", "go", 1, 50.0)


def test_normalize_text():
    assert normalize_text("  Yes! ") == "yes"
    assert normalize_text("Let's go.") == "let's go"
    assert normalize_text("...") == ""


def test_stratified_sample_is_balanced():
    y = np.array(WORDS * 30)
    idx = stratified_sample(y, per_word=5)
    assert len(idx) == 50
    assert all((y[idx] == w).sum() == 5 for w in WORDS)
