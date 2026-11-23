"""Tests for src/data.py — run with:  python -m pytest"""

import pytest

from src.data import _wanted, build_splits, check_no_speaker_leakage, speaker_id


def test_speaker_id():
    assert speaker_id("yes/0a7c2a8d_nohash_0.wav") == "0a7c2a8d"
    assert speaker_id("stop/ffd2ba2f_nohash_4.wav") == "ffd2ba2f"


def test_wanted_members():
    assert _wanted("./yes/abc_nohash_0.wav", ["yes"])
    assert _wanted("_background_noise_/white_noise.wav", ["yes"])
    assert _wanted("testing_list.txt", ["yes"])
    assert not _wanted("./cat/abc_nohash_0.wav", ["yes"])  # word not selected


def _make_fake_dataset(root):
    """Tiny fake dataset: 2 words, 3 speakers (one per split)."""
    for word in ["yes", "no"]:
        (root / word).mkdir()
        for spk in ["aaaa1111", "bbbb2222", "cccc3333"]:
            (root / word / f"{spk}_nohash_0.wav").write_bytes(b"")
    (root / "validation_list.txt").write_text("yes/bbbb2222_nohash_0.wav\nno/bbbb2222_nohash_0.wav\n")
    (root / "testing_list.txt").write_text("yes/cccc3333_nohash_0.wav\nno/cccc3333_nohash_0.wav\n")


def test_build_splits(tmp_path):
    _make_fake_dataset(tmp_path)
    splits = build_splits(tmp_path, words=["yes", "no"])

    assert len(splits["train"]) == 2 and len(splits["val"]) == 2 and len(splits["test"]) == 2
    assert {r["speaker"] for r in splits["train"]} == {"aaaa1111"}
    assert {r["speaker"] for r in splits["test"]} == {"cccc3333"}
    check_no_speaker_leakage(splits)  # must not raise


def test_leakage_is_detected():
    splits = {
        "train": [{"path": "yes/x_nohash_0.wav", "label": "yes", "speaker": "x"}],
        "val": [],
        "test": [{"path": "no/x_nohash_1.wav", "label": "no", "speaker": "x"}],
    }
    with pytest.raises(ValueError):
        check_no_speaker_leakage(splits)
