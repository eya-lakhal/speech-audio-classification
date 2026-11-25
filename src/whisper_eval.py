"""Bonus: how robust is an off-the-shelf ASR model (OpenAI Whisper) to noise?

Usage:  python -m src.whisper_eval [--model tiny] [--per-word 20]

A stratified sample of test clips (N per word) is corrupted with exactly the
same procedure as in noise_eval.py, transcribed by Whisper, and scored with
the Word Error Rate (jiwer). Because each reference is a single word, WER is
essentially the share of clips where Whisper did not output exactly that word
(it can exceed 100 % if Whisper hallucinates extra words).

We also report "keyword accuracy": the target word appears in the
transcription - a fairer comparison with the 10-class classifiers, which
cannot output anything else.
"""

from __future__ import annotations

import argparse
import re
import time

import matplotlib

matplotlib.use("Agg")
import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import whisper

from src.data import WORDS
from src.evaluate import RESULTS_DIR
from src.features import load_split_waveforms, make_noisy_set
from src.noise_eval import SNRS, snr_name


def normalize_text(text: str) -> str:
    """'  Yes! ' -> 'yes'  (lowercase, letters and spaces only)."""
    text = re.sub(r"[^a-z' ]", " ", text.lower())
    return " ".join(text.split())


def stratified_sample(y: np.ndarray, per_word: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(y == w), per_word, replace=False) for w in WORDS]
    return np.sort(np.concatenate(idx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "tiny.en", "base.en"])
    parser.add_argument("--per-word", type=int, default=20)
    args = parser.parse_args()

    x_test, y_test = load_split_waveforms("test")
    model = whisper.load_model(args.model, device="cpu")

    rows, transcripts = [], []
    for snr in SNRS:
        # Corrupt the FULL test set with the same seed as noise_eval.py, then
        # take the sample: the noisy clips are identical to the ones the
        # classifiers were scored on.
        x_noisy = x_test if snr is None else make_noisy_set(x_test, snr, seed=1234)
        idx = stratified_sample(y_test, args.per_word)
        refs, hyps = [], []
        t0 = time.time()
        for i in idx:
            # Whisper takes a 16 kHz float32 array directly (no ffmpeg needed)
            out = model.transcribe(x_noisy[i], language="en", fp16=False,
                                   temperature=0.0, condition_on_previous_text=False)
            hyp = normalize_text(out["text"])
            refs.append(y_test[i])
            hyps.append(hyp)
            transcripts.append({"snr": snr_name(snr), "reference": y_test[i], "hypothesis": hyp})
        # jiwer cannot score an empty hypothesis string against itself; an
        # empty output is a deletion, which "<empty>" scores identically.
        wer = jiwer.wer(refs, [h if h else "<empty>" for h in hyps])
        kw_acc = np.mean([r in h.split() for r, h in zip(refs, hyps)])
        exact = np.mean([r == h for r, h in zip(refs, hyps)])
        rows.append({"model": f"whisper-{args.model}", "snr": snr_name(snr), "wer": wer,
                     "keyword_accuracy": kw_acc, "exact_match": exact,
                     "n_clips": len(idx), "seconds": round(time.time() - t0)})
        print(f"{snr_name(snr):>6}  WER {wer:.3f}  keyword acc {kw_acc:.3f}  "
              f"exact {exact:.3f}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    tag = args.model.replace(".", "")
    df.to_csv(RESULTS_DIR / f"whisper_{tag}_results.csv", index=False)
    pd.DataFrame(transcripts).to_csv(RESULTS_DIR / f"whisper_{tag}_transcripts.csv", index=False)
    plot(tag)


def plot(_tag: str = "") -> None:
    """Plot WER for every whisper_*_results.csv found in results/."""
    order = [snr_name(s) for s in SNRS]
    fig, ax = plt.subplots(figsize=(6, 4))
    for csv in sorted(RESULTS_DIR.glob("whisper_*_results.csv")):
        df = pd.read_csv(csv).set_index("snr").loc[order]
        ax.plot(order, df["wer"] * 100, marker="o", label=df["model"].iloc[0])
    ax.set_title("Whisper word error rate vs SNR")
    ax.set_xlabel("Test condition (signal-to-noise ratio)")
    ax.set_ylabel("WER (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "whisper_wer_vs_snr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
