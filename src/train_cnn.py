"""CNN on log-mel spectrograms with waveform data augmentation.

Usage:
    python -m src.train_cnn                 # with augmentation (noise + time shift)
    python -m src.train_cnn --no-augment    # same model without augmentation

The model takes the raw 1-second waveform as input; the first layer computes
the log-mel spectrogram with tf.signal. Augmentation is therefore done on the
waveform (where "add noise" and "shift in time" are physically meaningful) and
the exact same feature computation is used at train and test time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import numpy as np
import tensorflow as tf
import keras

from src.data import WORDS
from src.evaluate import RESULTS_DIR, format_confusions, metrics, plot_confusion, top_confusions
from src.features import HOP, N_FFT, N_MELS, N_SAMPLES, SR, load_background_noises, load_split_waveforms

MODELS_DIR = Path("models")
SEED = 42


# --------------------------------------------------------------------------- #
# Log-mel layer
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="speech")
class LogMel(keras.layers.Layer):
    """waveform (batch, 16000) -> log-mel (batch, frames, N_MELS, 1)."""

    def __init__(self, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, **kwargs):
        super().__init__(**kwargs)
        self.sr, self.n_fft, self.hop, self.n_mels = sr, n_fft, hop, n_mels
        self.mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels, num_spectrogram_bins=n_fft // 2 + 1,
            sample_rate=sr, lower_edge_hertz=20.0, upper_edge_hertz=sr / 2)

    def call(self, waveform):
        stft = tf.signal.stft(waveform, frame_length=self.n_fft, frame_step=self.hop,
                              fft_length=self.n_fft, window_fn=tf.signal.hann_window)
        power = tf.square(tf.abs(stft))                      # (batch, frames, 257)
        mel = tf.matmul(power, self.mel_matrix)              # (batch, frames, 64)
        log_mel = tf.math.log(mel + 1e-6)
        return log_mel[..., tf.newaxis]                      # add a "channel" axis

    def get_config(self):
        return {**super().get_config(), "sr": self.sr, "n_fft": self.n_fft,
                "hop": self.hop, "n_mels": self.n_mels}


def build_model(n_classes: int = len(WORDS)) -> keras.Model:
    inputs = keras.Input(shape=(N_SAMPLES,), name="waveform")
    x = LogMel(name="log_mel")(inputs)
    # Learns a mean/std per mel band from the training data (like StandardScaler)
    x = keras.layers.Normalization(axis=2, name="mel_norm")(x)
    # 4 conv blocks; pooling halves time & frequency after each of the first 3.
    # Pooling early keeps the model cheap enough to train on a laptop CPU
    # (~45 M multiply-adds per clip).
    for i, filters in enumerate((32, 64, 128, 128)):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if i < 3:
            x = keras.layers.MaxPooling2D(2)(x)
            x = keras.layers.SpatialDropout2D(0.1)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="logmel_cnn")


# --------------------------------------------------------------------------- #
# Augmentation (runs inside tf.data, on the fly, different every epoch)
# --------------------------------------------------------------------------- #
def make_augment_fn(noise_bank: np.ndarray, p_noise=0.8, snr_range=(0.0, 20.0),
                    max_shift=1600):
    """Returns f(waveform, label) that randomly shifts and adds noise.

    * time shift: roll by up to +-100 ms and zero the wrapped part
      (the word is not always centred in real life)
    * noise: with prob. 0.8, add a random excerpt of the *training part* of the
      background noises at a random SNR in [0, 20] dB
    """
    bank = tf.constant(noise_bank, dtype=tf.float32)

    def augment(wave, label):
        # --- time shift
        shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
        wave = tf.roll(wave, shift, axis=0)
        idx = tf.range(N_SAMPLES)
        keep = tf.logical_or(tf.logical_and(shift >= 0, idx >= shift),
                             tf.logical_and(shift < 0, idx < N_SAMPLES + shift))
        wave = tf.where(keep, wave, 0.0)

        # --- background noise at random SNR
        def add_noise(w):
            start = tf.random.uniform([], 0, tf.shape(bank)[0] - N_SAMPLES, dtype=tf.int32)
            noise = bank[start : start + N_SAMPLES]
            snr = tf.random.uniform([], *snr_range)
            p_clean = tf.reduce_mean(tf.square(w))
            p_noise = tf.reduce_mean(tf.square(noise)) + 1e-12
            scale = tf.sqrt(p_clean / (p_noise * tf.pow(10.0, snr / 10.0)))
            mixed = w + scale * noise
            return mixed / (tf.reduce_max(tf.abs(mixed)) + 1e-8)

        wave = tf.cond(tf.random.uniform([]) < p_noise, lambda: add_noise(wave), lambda: wave)
        return wave, label

    return augment


def make_dataset(x, y, batch=128, shuffle=False, augment_fn=None):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def encode(labels) -> np.ndarray:
    return np.array([WORDS.index(l) for l in labels], dtype=np.int32)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    name = "cnn_noaug" if args.no_augment else "cnn"

    keras.utils.set_random_seed(SEED)
    x_tr, y_tr = load_split_waveforms("train")
    x_va, y_va = load_split_waveforms("val")
    x_te, y_te = load_split_waveforms("test")
    y_tr_i, y_va_i = encode(y_tr), encode(y_va)

    model = build_model()
    # Fit the Normalization layer on log-mels of (a sample of) clean training clips
    logmel = model.get_layer("log_mel")
    sample = logmel(tf.constant(x_tr[:: 10]))
    model.get_layer("mel_norm").adapt(sample)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    augment_fn = None
    if not args.no_augment:
        noise_bank = np.concatenate(load_background_noises("train"))
        augment_fn = make_augment_fn(noise_bank)

    train_ds = make_dataset(x_tr, y_tr_i, shuffle=True, augment_fn=augment_fn)
    val_ds = make_dataset(x_va, y_va_i)

    MODELS_DIR.mkdir(exist_ok=True)
    # If training is interrupted (sleep, crash, closed terminal), running the
    # same command again resumes from the last finished epoch.
    backup_dir = MODELS_DIR / f"backup_{name}"
    history_csv = MODELS_DIR / f"history_{name}.csv"
    if not backup_dir.exists() and history_csv.exists():
        history_csv.unlink()  # fresh start -> fresh history
    callbacks = [
        keras.callbacks.BackupAndRestore(str(backup_dir)),
        keras.callbacks.CSVLogger(str(history_csv), append=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=callbacks, verbose=2)
    elapsed = time.time() - t0
    import pandas as pd
    history = pd.read_csv(history_csv).to_dict(orient="list")  # all epochs, even across resumes
    model.save(MODELS_DIR / f"{name}.keras")

    probs = model.predict(make_dataset(x_te, encode(y_te)), verbose=0)
    y_pred = np.array(WORDS)[probs.argmax(axis=1)]
    test = metrics(y_te, y_pred)
    conf = top_confusions(y_te, y_pred)
    print(f"\n{name}: test accuracy {test['accuracy']:.4f}  test F1 {test['f1_macro']:.4f}"
          f"  ({elapsed / 60:.1f} min)")
    print("Most frequent confusions:\n" + format_confusions(conf))

    title = "CNN (augmented)" if name == "cnn" else "CNN (no augmentation)"
    plot_confusion(y_te, y_pred, f"{title} - clean test", RESULTS_DIR / f"confusion_{name}.png")
    plot_history(history, title, RESULTS_DIR / f"history_{name}.png")
    with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
        json.dump({"test": test, "top_confusions": conf, "epochs": len(history["loss"]),
                   "train_minutes": round(elapsed / 60, 1),
                   "params": int(model.count_params())}, f, indent=2)


def plot_history(h: dict, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="val")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
