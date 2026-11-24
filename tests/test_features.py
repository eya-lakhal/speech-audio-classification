"""Tests for src/features.py."""

import numpy as np
import pytest

from src.features import (
    CLASSIC_FEATURE_NAMES, N_MELS, N_SAMPLES, SR,
    add_noise_snr, classic_features, log_mel, normalize, pad_or_trim,
)


def _tone(n=N_SAMPLES, freq=440.0):
    t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pad_short_clip():
    y = pad_or_trim(np.ones(8000, dtype=np.float32))
    assert len(y) == N_SAMPLES
    assert y[0] == 0 and y[-1] == 0 and y[N_SAMPLES // 2] == 1  # centred


def test_trim_long_clip():
    assert len(pad_or_trim(np.ones(20000, dtype=np.float32))) == N_SAMPLES


def test_normalize_peak_and_silence():
    assert np.isclose(np.max(np.abs(normalize(_tone()))), 1.0)
    silent = np.zeros(100, dtype=np.float32)
    assert np.all(normalize(silent) == 0)  # no division by zero


@pytest.mark.parametrize("snr", [20, 10, 5, 0])
def test_add_noise_reaches_target_snr(snr):
    rng = np.random.default_rng(0)
    clean = _tone()
    noise = rng.standard_normal(N_SAMPLES).astype(np.float32)
    mixed = add_noise_snr(clean, noise, snr)
    # mixed = g * (clean + noise_scaled); estimate g by projecting onto clean
    # (noise is ~uncorrelated with the tone), then recover the added noise.
    gain = np.dot(mixed, clean) / np.dot(clean, clean)
    residual = mixed / gain - clean
    measured = 10 * np.log10(np.mean(clean ** 2) / np.mean(residual ** 2))
    assert abs(measured - snr) < 0.5, measured


def test_classic_features_shape():
    f = classic_features(_tone())
    assert f.shape == (len(CLASSIC_FEATURE_NAMES),) == (84,)
    assert np.all(np.isfinite(f))


def test_log_mel_shape():
    assert log_mel(_tone()).shape[0] == N_MELS
