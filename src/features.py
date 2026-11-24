"""Audio preprocessing, noise mixing and feature extraction.

Pipeline for every clip:
    load -> mono -> resample to 16 kHz -> pad/trim to exactly 1 s -> peak-normalise

Features:
    * classical: MFCC (+ delta, delta-delta), spectral centroid, zero-crossing
      rate and RMS, each summarised over time by its mean and std -> 84 values
    * deep learning: log-mel spectrogram (64 mel bands x 98 frames)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

SR = 16_000            # target sample rate (Hz)
N_SAMPLES = SR         # 1 second
N_FFT = 512            # 32 ms analysis window
HOP = 160              # 10 ms hop -> 101 frames per second with librosa centering
N_MFCC = 13
N_MELS = 64

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "speech_commands"
CACHE_DIR = DATA_DIR / "cache"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Read a wav file as float32 mono at `sr` Hz."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:                       # stereo -> mono
        y = y.mean(axis=1)
    if file_sr != sr:                    # Speech Commands is already 16 kHz,
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)  # but be safe
    return y.astype(np.float32)


def pad_or_trim(y: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Zero-pad (centred) or crop (centred) to exactly `n` samples."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start : start + n]
    missing = n - len(y)
    left = missing // 2
    return np.pad(y, (left, missing - left))


def normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak normalisation: the loudest sample becomes +-1.

    Speakers recorded themselves with very different microphones and volumes;
    this removes the loudness difference so models focus on the content.
    """
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > eps else y.astype(np.float32)


def preprocess(path: str | Path) -> np.ndarray:
    return normalize(pad_or_trim(load_audio(path)))


# --------------------------------------------------------------------------- #
# Waveform cache (decode the ~38k wav files only once)
# --------------------------------------------------------------------------- #
def load_split_waveforms(split: str, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return (waveforms [N, 16000] float32, labels [N] str) for a split.

    Stored as int16 in data/cache/<split>.npz (half the size of float32).
    """
    from src.data import load_split  # local import keeps this module light

    cache = CACHE_DIR / f"{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"].astype(np.float32) / 32767.0, d["y"]

    rows = load_split(split, DATA_DIR)
    paths = [DATASET_DIR / r["path"] for r in rows]
    waves = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(preprocess)(p) for p in tqdm(paths, desc=f"Loading {split}")
    )
    x = np.stack(waves)
    y = np.array([r["label"] for r in rows])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=(x * 32767).astype(np.int16), y=y)
    return x, y


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
NOISE_TRAIN_FRACTION = 0.7  # first 70 % of each noise file: training augmentation
                            # last 30 %: test-time corruption (no noise leakage)


def load_background_noises(part: str = "test") -> list[np.ndarray]:
    """Load the 6 `_background_noise_` recordings, keeping only one part.

    part="train" -> first 70 % of every file, part="test" -> last 30 %.
    Using disjoint parts means the CNN never hears, during training, the exact
    noise excerpts it is evaluated on.
    """
    noises = []
    for f in sorted((DATASET_DIR / "_background_noise_").glob("*.wav")):
        y = load_audio(f)
        cut = int(len(y) * NOISE_TRAIN_FRACTION)
        noises.append(y[:cut] if part == "train" else y[cut:])
    return noises


def random_noise_segment(noises: list[np.ndarray], rng: np.random.Generator,
                         n: int = N_SAMPLES) -> np.ndarray:
    noise = noises[rng.integers(len(noises))]
    start = rng.integers(0, len(noise) - n)
    return noise[start : start + n]


def add_noise_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix `noise` into `clean` so that 10*log10(P_clean / P_noise) == snr_db.

    P = mean(x**2) is the signal power. We rescale the noise by
    sqrt(P_clean / (P_noise * 10**(snr/10))), then peak-normalise the mixture
    (the same normalisation every clean clip received).
    """
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return normalize(clean + scale * noise)


def make_noisy_set(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Corrupt every clip of `x` with a random test-noise excerpt at `snr_db`."""
    rng = np.random.default_rng(seed)
    noises = load_background_noises("test")
    return np.stack([add_noise_snr(c, random_noise_segment(noises, rng), snr_db) for c in x])


# --------------------------------------------------------------------------- #
# Classical features
# --------------------------------------------------------------------------- #
CLASSIC_FEATURE_NAMES = (
    [f"{stat}_{name}{i}" for name in ("mfcc", "dmfcc", "ddmfcc")
     for i in range(N_MFCC) for stat in ("mean", "std")]
    + [f"{stat}_{name}" for name in ("centroid", "zcr", "rms") for stat in ("mean", "std")]
)


def classic_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """84-dim vector: mean & std over time of 39 MFCC-based + 3 spectral descriptors."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP, n_mels=40)
    d1 = librosa.feature.delta(mfcc)            # how MFCCs change (speed)
    d2 = librosa.feature.delta(mfcc, order=2)   # acceleration
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)

    feats = []
    for m in (mfcc, d1, d2):
        # interleave mean/std per coefficient, matching CLASSIC_FEATURE_NAMES
        feats.append(np.stack([m.mean(axis=1), m.std(axis=1)], axis=1).ravel())
    for m in (centroid / 1000.0, zcr, rms):     # centroid in kHz keeps scales close
        feats.append(np.array([m.mean(), m.std()]))
    return np.concatenate(feats).astype(np.float32)


def classic_features_batch(x: np.ndarray, n_jobs: int = -1, desc: str = "Features") -> np.ndarray:
    out = Parallel(n_jobs=n_jobs, batch_size=128)(
        delayed(classic_features)(y) for y in tqdm(x, desc=desc)
    )
    return np.stack(out)


# --------------------------------------------------------------------------- #
# Log-mel (librosa version, used for exploration / figures)
# --------------------------------------------------------------------------- #
def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (N_MELS, frames)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmin=20, fmax=sr // 2)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0)
