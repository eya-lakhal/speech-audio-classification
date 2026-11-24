"""Download Google Speech Commands v2 and build the official train/val/test split.

Usage (from the project root):
    python -m src.data                 # download + extract + split
    python -m src.data --skip-download # only rebuild the split CSVs

Output:
    data/speech_commands/<word>/*.wav        audio clips for the 10 target words
    data/speech_commands/_background_noise_/ long noise recordings (used in step 5)
    data/splits/{train,val,test}.csv         columns: path,label,speaker
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

from tqdm import tqdm

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# The 10 keywords of the classic "Speech Commands" benchmark.
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Files we also need from the archive besides the word folders.
EXTRA_MEMBERS = ("validation_list.txt", "testing_list.txt", "LICENSE", "README.md")
NOISE_DIR = "_background_noise_"

DEFAULT_DATA_DIR = Path("data")


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, max_retries: int = 50) -> Path:
    """Download `url` to `dest` with a progress bar. Skips if the file exists.

    The file is written to `<dest>.part` and only renamed at the end, so an
    interrupted download never looks complete. If the connection drops, the
    download *resumes* where it stopped (HTTP "Range" header) instead of
    starting again from zero - important for a 2.4 GB file on a slow network.
    """
    if dest.exists():
        print(f"[download] {dest} already exists, skipping.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers.get("Content-Length", 0))

    bar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
    for attempt in range(max_retries):
        done = tmp.stat().st_size if tmp.exists() else 0
        bar.n = done
        bar.refresh()
        if total and done >= total:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "ab") as f:
                if done and response.status != 206:   # server ignored Range:
                    f.truncate(0)                       # restart from scratch
                    bar.n = 0
                while chunk := response.read(1 << 20):  # 1 MB at a time
                    f.write(chunk)
                    bar.update(len(chunk))
            if not total or tmp.stat().st_size >= total:
                break
        except OSError as e:  # connection reset, timeout, ...
            print(f"\n[download] interrupted ({e}); resuming (attempt {attempt + 2})")
            time.sleep(5)
    bar.close()
    if total and tmp.stat().st_size != total:
        raise RuntimeError("Download incomplete, run the script again to resume.")

    tmp.rename(dest)  # only "finalise" the file once fully downloaded
    return dest


# --------------------------------------------------------------------------- #
# 2. Extract only what we need
# --------------------------------------------------------------------------- #
def _wanted(name: str, words: list[str]) -> bool:
    """Should this archive member be extracted?"""
    name = name.lstrip("./")
    top = name.split("/", 1)[0]
    return top in words or top == NOISE_DIR or name in EXTRA_MEMBERS


def extract_subset(archive: Path, out_dir: Path, words: list[str] = WORDS) -> None:
    """Extract the target words + background noise + official split lists."""
    if (out_dir / "testing_list.txt").exists():
        print(f"[extract] {out_dir} already extracted, skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # "r|gz" reads the archive as a stream: fast and low-memory.
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tqdm(tar, desc="Extracting", unit=" files"):
            if member.isfile() and _wanted(member.name, words):
                # filter="data" blocks malicious paths (../, absolute paths...)
                tar.extract(member, out_dir, filter="data")


# --------------------------------------------------------------------------- #
# 3. Official split (no speaker leakage)
# --------------------------------------------------------------------------- #
def speaker_id(rel_path: str) -> str:
    """'yes/0a7c2a8d_nohash_0.wav' -> '0a7c2a8d' (the anonymous speaker hash)."""
    return Path(rel_path).name.split("_nohash_")[0]


def _read_list(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_splits(dataset_dir: Path, words: list[str] = WORDS) -> dict[str, list[dict]]:
    """Assign every clip to train/val/test using the dataset's official lists.

    validation_list.txt and testing_list.txt are given by the dataset authors.
    Every clip not listed in either goes to train. The authors assigned clips
    by hashing the *speaker id*, so one speaker is always in a single split.
    """
    val_set = _read_list(dataset_dir / "validation_list.txt")
    test_set = _read_list(dataset_dir / "testing_list.txt")

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for word in words:
        for wav in sorted((dataset_dir / word).glob("*.wav")):
            rel = f"{word}/{wav.name}"  # same format as in the official lists
            split = "val" if rel in val_set else "test" if rel in test_set else "train"
            splits[split].append({"path": rel, "label": word, "speaker": speaker_id(rel)})
    return splits


def check_no_speaker_leakage(splits: dict[str, list[dict]]) -> None:
    """Raise an error if any speaker appears in more than one split."""
    speakers = {name: {row["speaker"] for row in rows} for name, rows in splits.items()}
    names = list(speakers)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = speakers[a] & speakers[b]
            if overlap:
                raise ValueError(
                    f"{len(overlap)} speakers shared between {a} and {b}, "
                    f"e.g. {sorted(overlap)[:3]}"
                )


def save_splits(splits: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "speaker"])
            writer.writeheader()
            writer.writerows(rows)


def load_split(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Read data/splits/<name>.csv back (used by the next steps)."""
    with open(data_dir / "splits" / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_summary(splits: dict[str, list[dict]]) -> None:
    print(f"\n{'split':<6} {'clips':>7} {'speakers':>9}")
    for name, rows in splits.items():
        print(f"{name:<6} {len(rows):>7} {len({r['speaker'] for r in rows}):>9}")
    print("\nClips per word (train / val / test):")
    counts = {name: Counter(r["label"] for r in rows) for name, rows in splits.items()}
    for w in WORDS:
        print(f"  {w:<6} {counts['train'][w]:>5} / {counts['val'][w]:>4} / {counts['test'][w]:>4}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.data_dir / "speech_commands"
    if not args.skip_download:
        archive = download(DATASET_URL, args.data_dir / "speech_commands_v0.02.tar.gz")
        extract_subset(archive, dataset_dir)

    splits = build_splits(dataset_dir)
    check_no_speaker_leakage(splits)
    save_splits(splits, args.data_dir / "splits")
    print_summary(splits)
    print("\nNo speaker leakage between splits. CSVs written to", args.data_dir / "splits")


if __name__ == "__main__":
    main()
